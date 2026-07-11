#!/usr/bin/env python3
"""
Pipeline de restauration audio pour causeries du Dhamma — nettoyer.py
==================================================================================
Nettoie des enregistrements de parole (causeries, conférences, méditations
guidées) avec un objectif de QUALITÉ EXTRÊME et de TOUCHER LÉGER : réduction
du bruit de fond sans dénaturer la voix ni le silence, normalisation du
volume sans écrasement dynamique ni clipping.

Philosophie (issue d'une revue des meilleures pratiques, juillet 2026) :
  - Le silence fait partie du contenu : jamais de gate, jamais de silence
    numérique absolu. Le fond de salle résiduel est conservé volontairement.
  - Une seule passe de débruitage, avec mélange dry/wet spectral intégré
    (limite d'atténuation DeepFilterNet) — pas de chaînage de débruiteurs.
  - Normalisation LINÉAIRE (gain constant sur tout le fichier) : aucun
    limiteur, aucune compression par défaut. Le limiteur suréchantillonné
    n'intervient qu'en dernier recours (ex. cloches de méditation qui
    empêcheraient d'atteindre la cible) et son usage est signalé.
  - Nivelage dynamique intra-fichier OPTIONNEL (--niveler) : tout niveleur
    aveugle risque de manger les fins de phrases ou de gonfler les
    respirations — désactivé par défaut, réglages très doux si activé.
  - Contrôle qualité automatique : DNSMOS (métrique perceptuelle Microsoft)
    avant/après sur fenêtres de parole. Si la qualité vocale (SIG) baisse,
    le fichier est signalé pour écoute manuelle — jamais de confiance
    aveugle dans le traitement.

Architecture en 7 passes :
  1. Analyse            → ffprobe + détection de ronflette secteur (50/60 Hz
                          et harmoniques) par densité spectrale de puissance
  2. Conditionnement    → décodage 48 kHz float mono (soxr), passe-haut 60 Hz,
                          notchs étroits UNIQUEMENT sur les harmoniques de
                          ronflette réellement détectées
  3. Débruitage         → moteur au choix (--moteur, défaut : auto) :
                          · dfn         DeepFilterNet3 (48 kHz natif) par tronçons
                                        de 60 s avec recouvrement et fondu enchaîné ;
                                        limite d'atténuation = mélange dry/wet
                                        spectral (15 dB ≈ 18 % d'original conservé)
                          · mossformer2 MossFormer2_SE_48K (ClearerVoice-Studio),
                                        via pont dans l'env conda `clearvoice` —
                                        plus fort, préserve mieux certaines voix
                                        que DFN abîme (validé sur matériel réel) ;
                                        contrepartie mesurée : comprime la
                                        dynamique (~8 dB de crête) et son gain
                                        n'est pas strictement constant
                          · afftdn      soustraction spectrale classique ffmpeg,
                                        très douce — voix quasi intacte, réduction
                                        modérée (l'esprit RX, sans profil manuel)
                          · auto        DFN d'abord ; si le contrôle DNSMOS révèle
                                        une voix abîmée (SIG -0.10), bascule sur
                                        MossFormer2 (ou afftdn) et garde le meilleur
  4. Nivelage (option)  → dynaudnorm très doux (fenêtre ~15 s, gain max 8x,
                          seuil anti-amplification des silences)
  5. Normalisation      → mesure BS.1770 (pyloudnorm) puis GAIN CONSTANT
                          vers -19 LUFS (mono), plafond -1.5 dBTP vérifié
                          par suréchantillonnage 4x ; si la cible est
                          inatteignable : gain réduit (≤2 dB d'écart) ou
                          limiteur 192 kHz sur les seules crêtes (cloches)
  6. Encodage           → MP3 LAME V0 mono 48 kHz, tags ID3 copiés de la
                          source, ID3v2.3 (compatibilité maximale)
  7. Contrôle qualité   → DNSMOS avant/après (SIG/BAK/OVRL), vérification
                          LUFS + crête vraie du MP3 final, durée identique ;
                          rapport JSON à côté du fichier de sortie

Mode hybride iZotope RX :
  --exporter-rx         → s'arrête après la passe 2 et écrit
                          <nom>_pour_rx.wav (48 kHz float) ; débruitez-le
                          dans RX (réduction spectrale), puis reprenez avec :
  --importer-rx F.wav   → saute la passe 3 et reprend le pipeline (nivelage,
                          normalisation, encodage, QC) sur le fichier RX

Usage :
  python nettoyer.py causerie.mp3                    # un fichier
  python nettoyer.py dossier/                        # tous les mp3/wav/m4a/flac
  python nettoyer.py causerie.mp3 --moteur mossformer2  # imposer un moteur
  python nettoyer.py causerie.mp3 --reduction 20     # débruitage dfn plus fort
  python nettoyer.py causerie.mp3 --reduction 10     # plus de fond conservé
  python nettoyer.py causerie.mp3 --niveler          # nivelage doux intra-fichier
  python nettoyer.py causerie.mp3 --lufs -16         # cible plus forte
  python nettoyer.py causerie.mp3 --exporter-rx      # WAV pour iZotope RX
  python nettoyer.py causerie.mp3 --importer-rx causerie_rx.wav
  python nettoyer.py dossier/ -o propre/             # dossier de sortie
  python nettoyer.py causerie.mp3 --cpu              # sans GPU

Prérequis (env conda `traduction-amd`) :
  pip install deepfilternet pyloudnorm speechmos
  # + ffmpeg installé (loudnorm/ebur128/dynaudnorm/alimiter)

⚠️ AMD : non testé sur ROCm. DeepFilterNet est du torch pur → devrait
fonctionner via PyTorch-HIP (qui se présente comme "cuda") ; DNSMOS
(speechmos/onnxruntime) tourne sur CPU ; le moteur mossformer2
(env `clearvoice`, torch CUDA) est incertain sur ROCm — en cas de souci,
--moteur dfn ou afftdn, et --cpu reste toujours disponible.
"""

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# ── Constantes issues de la revue des meilleures pratiques ──────────────────
SR_TRAVAIL = 48000          # taux natif de DeepFilterNet3
PASSE_HAUT_HZ = 60          # sous le fondamental d'une voix masculine grave.
                            # Validé par DNSMOS sur matériel réel : à 75 Hz la
                            # qualité vocale (SIG) baisse déjà mesurablement,
                            # alors que le gain anti-bruit (BAK) plafonne dès 50 Hz
PASSE_HAUT_POLES = 2        # 12 dB/octave — pente douce, pas de phase brutale

REDUCTION_DB = 15           # limite d'atténuation DFN (mélange dry/wet) :
                            # 10 dB ≈ 32% dry, 15 dB ≈ 18%, 20 dB ≈ 10%
TRONCON_S = 60.0            # débruitage par tronçons (l'API DFN charge tout
RECOUVREMENT_S = 1.0        # en VRAM sinon — plante sur 1 h d'audio)

AFFTDN_NR = 12              # réduction afftdn (dB) — au-delà, le suiveur de
                            # bruit plafonne de toute façon (mesuré)
CLEARVOICE_PYTHON = Path.home() / "miniconda3/envs/clearvoice/bin/python"
PONT_MOSSFORMER = Path(__file__).resolve().parent / "mossformer_bridge.py"

LUFS_CIBLE_MONO = -19.0     # ≈ -16 LUFS stéréo perçu (convention podcast mono)
LUFS_CIBLE_STEREO = -16.0   # spec Apple Podcasts
TP_PLAFOND_DB = -1.5        # marge pour le dépassement de crête à l'encodage MP3
GAIN_MAX_DB = 24.0          # garde-fou : fichier quasi muet → pas de +50 dB
EXCES_GAIN_REDUIT_DB = 2.0  # écart à la cible accepté avant de sortir le limiteur

# Nivelage optionnel — réglages "toucher léger" (fenêtre ~15.5 s, gain max 8x,
# seuil t pour ne jamais amplifier le fond de salle pendant les silences)
DYNAUDNORM_DOUX = "dynaudnorm=f=500:g=31:p=0.9:m=8:t=0.02:o=0.3:b=1"

MP3_QUALITE_VBR = 0         # LAME V0 — transparent pour la parole, avec marge

# Détection de ronflette : fondamentaux (50 Hz UE, 60 Hz US) et harmoniques —
# le passe-haut à 60 Hz n'atténue que faiblement un ronflement à 50 Hz,
# d'où la détection dès le fondamental
HUM_HARMONIQUES = [50, 60, 100, 150, 200, 250, 300, 120, 180, 240]
HUM_SEUIL_DB = 8.0          # proéminence minimale au-dessus du plancher local
HUM_LARGEUR_HZ = 4.0        # largeur du notch — étroit pour épargner la voix

# Contrôle qualité DNSMOS
QC_FENETRES = 8             # nombre de fenêtres de parole échantillonnées
QC_FENETRE_S = 9.0          # durée d'une fenêtre (format d'entrée DNSMOS)
QC_SIG_BAISSE_MAX = 0.10    # au-delà → la voix a été abîmée → alerte
EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma"}


# ─────────────────────────────────────────────────────────────────────────────
# Utilitaires
# ─────────────────────────────────────────────────────────────────────────────

def executer(cmd, description=""):
    """Exécute une commande, renvoie stderr (ffmpeg y écrit tout)."""
    resultat = subprocess.run(cmd, capture_output=True, text=True)
    if resultat.returncode != 0:
        raise RuntimeError(f"Échec {description or cmd[0]} :\n{resultat.stderr[-2000:]}")
    return resultat.stderr


def db_vers_lineaire(db):
    return 10 ** (db / 20.0)


def verifier_dependances(args):
    manquants = []
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        manquants.append("ffmpeg/ffprobe (binaire système)")
    try:
        import pyloudnorm  # noqa: F401
    except ImportError:
        manquants.append("pyloudnorm (pip install pyloudnorm)")
    if not args.importer_rx and not args.exporter_rx:
        if args.moteur in ("auto", "dfn"):
            try:
                import df  # noqa: F401
            except ImportError:
                manquants.append("deepfilternet (pip install deepfilternet)")
        if args.moteur == "mossformer2" and not CLEARVOICE_PYTHON.exists():
            manquants.append("env conda `clearvoice` (voir mossformer_bridge.py)")
    try:
        import speechmos  # noqa: F401
    except ImportError:
        print("⚠ speechmos absent — contrôle qualité DNSMOS désactivé "
              "(pip install speechmos)")
    if manquants:
        print("Dépendances manquantes :")
        for m in manquants:
            print(f"  - {m}")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Passe 1 — Analyse
# ─────────────────────────────────────────────────────────────────────────────

def analyser_source(fichier):
    """ffprobe : format, durée, canaux, taux d'échantillonnage."""
    sortie = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=sample_rate,channels,bit_rate:format=duration",
         "-of", "json", str(fichier)],
        capture_output=True, text=True)
    infos = json.loads(sortie.stdout)
    flux = infos["streams"][0]
    return {
        "sr": int(flux.get("sample_rate", 0)),
        "canaux": int(flux.get("channels", 1)),
        "debit": int(flux["bit_rate"]) if flux.get("bit_rate", "").isdigit() else None,
        "duree": float(infos["format"]["duration"]),
    }


def detecter_ronflette(wav_48k):
    """Cherche des harmoniques de ronflette secteur par densité spectrale.

    Welch sur ~10 min au centre du fichier, résolution < 0.4 Hz. Une
    harmonique n'est retenue que si elle dépasse nettement (HUM_SEUIL_DB)
    le plancher spectral local — on ne notche jamais 'au cas où', chaque
    notch retire aussi du signal.
    """
    from scipy.signal import welch
    import soundfile as sf

    infos = sf.info(str(wav_48k))
    centre = infos.frames // 2
    largeur = min(infos.frames, 10 * 60 * infos.samplerate)
    debut = max(0, centre - largeur // 2)
    audio, sr = sf.read(str(wav_48k), start=debut, frames=largeur, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    freqs, psd = welch(audio, fs=sr, nperseg=2 ** 17)
    psd_db = 10 * np.log10(psd + 1e-20)

    detectees = []
    for f0 in HUM_HARMONIQUES:
        bande = (freqs > f0 - 1.5) & (freqs < f0 + 1.5)
        voisinage = ((freqs > f0 - 15) & (freqs < f0 + 15)) & ~((freqs > f0 - 3) & (freqs < f0 + 3))
        if not bande.any() or not voisinage.any():
            continue
        proeminence = psd_db[bande].max() - np.median(psd_db[voisinage])
        if proeminence >= HUM_SEUIL_DB:
            f_exacte = freqs[bande][psd_db[bande].argmax()]
            detectees.append(round(float(f_exacte), 1))
    return detectees


# ─────────────────────────────────────────────────────────────────────────────
# Passe 2 — Conditionnement (décodage 48 kHz + passe-haut + notchs)
# ─────────────────────────────────────────────────────────────────────────────

def decoder(fichier, wav_sortie, canaux):
    """Décode vers WAV float 48 kHz via soxr (rééchantillonnage haute qualité)."""
    executer(
        ["ffmpeg", "-y", "-hide_banner", "-i", str(fichier),
         "-map", "0:a:0", "-vn",
         "-af", f"aresample=resampler=soxr:precision=28:out_sample_rate={SR_TRAVAIL}",
         "-ac", str(canaux), "-c:a", "pcm_f32le", str(wav_sortie)],
        "décodage 48 kHz")


def conditionner(wav_entree, wav_sortie, notchs, passe_haut_hz):
    """Passe-haut doux + notchs étroits sur les harmoniques détectées."""
    filtres = []
    if passe_haut_hz > 0:
        filtres.append(f"highpass=f={passe_haut_hz}:poles={PASSE_HAUT_POLES}")
    for f0 in notchs:
        filtres.append(f"bandreject=f={f0}:width_type=h:w={HUM_LARGEUR_HZ}")
    if not filtres:
        shutil.copy(str(wav_entree), str(wav_sortie))
        return
    executer(
        ["ffmpeg", "-y", "-hide_banner", "-i", str(wav_entree),
         "-af", ",".join(filtres), "-c:a", "pcm_f32le", str(wav_sortie)],
        "conditionnement")


# ─────────────────────────────────────────────────────────────────────────────
# Passe 3 — Débruitage DeepFilterNet3 par tronçons
# ─────────────────────────────────────────────────────────────────────────────

_dfn_cache = None


def charger_dfn(sur_cpu):
    """Charge DeepFilterNet3 une seule fois pour tout le lot."""
    global _dfn_cache
    if _dfn_cache is None:
        os.environ.setdefault("DF_LOG_LEVEL", "error")
        import warnings
        warnings.filterwarnings("ignore", message=".*AudioMetaData.*")
        import torch
        from df.enhance import init_df
        if sur_cpu:
            torch.cuda.is_available = lambda: False  # force CPU
        modele, etat, _ = init_df(log_level="error")
        _dfn_cache = (modele, etat)
    return _dfn_cache


def debruiter_dfn(wav_entree, wav_sortie, reduction_db, sur_cpu):
    """DFN3 par tronçons avec recouvrement et fondu enchaîné (overlap-add).

    L'API Python de DFN traite le fichier entier d'un bloc (20 Go de VRAM
    pour 1 h d'audio — issue GitHub #542, wontfix). On tronçonne donc à
    60 s avec 1 s de recouvrement : DFN est causal (40 ms d'anticipation),
    1 s de contexte suffit largement, et les rampes complémentaires du
    fondu sommant à 1, le niveau est strictement préservé.

    `atten_lim_db` est le mélange dry/wet officiel de DFN, appliqué dans
    le domaine spectral complexe (cohérent en phase — pas de filtrage en
    peigne, contrairement à un mélange temporel de deux fichiers).
    """
    import soundfile as sf
    import torch
    from df.enhance import enhance

    modele, etat = charger_dfn(sur_cpu)
    audio, sr = sf.read(str(wav_entree), dtype="float32", always_2d=True)
    audio = audio.T  # (canaux, n)
    n = audio.shape[1]
    troncon = int(TRONCON_S * sr)
    recouvrement = int(RECOUVREMENT_S * sr)

    sortie = np.zeros_like(audio)
    poids = np.zeros(n, dtype=np.float32)

    for debut in range(0, n, troncon):
        a = max(0, debut - recouvrement)
        b = min(n, debut + troncon + recouvrement)
        bloc = torch.from_numpy(audio[:, a:b])
        with torch.no_grad():
            propre = enhance(modele, etat, bloc, pad=True,
                             atten_lim_db=reduction_db).cpu().numpy()
        propre = propre[:, :b - a]  # sécurité si enhance rallonge d'un échantillon

        rampe = np.ones(b - a, dtype=np.float32)
        if a > 0:
            rampe[:recouvrement] = np.linspace(0, 1, recouvrement, dtype=np.float32)
        if b < n:
            rampe[-recouvrement:] = np.linspace(1, 0, recouvrement, dtype=np.float32)
        sortie[:, a:b] += propre * rampe
        poids[a:b] += rampe

    sortie /= np.maximum(poids, 1e-8)
    sf.write(str(wav_sortie), sortie.T, sr, subtype="FLOAT")
    if not sur_cpu:
        torch.cuda.empty_cache()


_pont_mf2 = None


def pont_mossformer2():
    """Démarre (une fois pour tout le lot) le pont ClearerVoice-Studio."""
    global _pont_mf2
    if _pont_mf2 is None:
        if not CLEARVOICE_PYTHON.exists():
            raise RuntimeError(
                "moteur mossformer2 indisponible — créer l'environnement :\n"
                "  conda create -n clearvoice python=3.10 -y\n"
                f"  {CLEARVOICE_PYTHON} -m pip install clearvoice")
        proc = subprocess.Popen(
            [str(CLEARVOICE_PYTHON), str(PONT_MOSSFORMER)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True,
            cwd=str(PONT_MOSSFORMER.parent))  # cwd stable → cache modèle unique
        _pont_mf2 = proc
        pont_commander({"cmd": "init"})
    return _pont_mf2


def pont_commander(requete):
    proc = _pont_mf2
    proc.stdin.write(json.dumps(requete) + "\n")
    proc.stdin.flush()
    ligne = proc.stdout.readline()
    if not ligne:
        raise RuntimeError("le pont MossFormer2 s'est arrêté de façon inattendue")
    reponse = json.loads(ligne)
    if not reponse.get("ok"):
        raise RuntimeError(f"pont MossFormer2 : {reponse.get('erreur')}")
    return reponse


def fermer_pont():
    global _pont_mf2
    if _pont_mf2 is not None:
        try:
            pont_commander({"cmd": "quit"})
            _pont_mf2.wait(timeout=10)
        except Exception:
            _pont_mf2.kill()
        _pont_mf2 = None


def debruiter_mossformer2(wav_entree, wav_sortie):
    """MossFormer2_SE_48K via le pont (ClearerVoice segmente lui-même les
    fichiers longs en interne, mémoire constante — validé sur 60 min)."""
    pont_mossformer2()
    pont_commander({"cmd": "traiter", "entree": str(wav_entree),
                    "sortie": str(wav_sortie)})


def debruiter_afftdn(wav_entree, wav_sortie):
    """Soustraction spectrale classique avec suiveur de bruit — le moteur le
    plus doux pour la voix, réduction modérée du fond."""
    executer(
        ["ffmpeg", "-y", "-hide_banner", "-i", str(wav_entree),
         "-af", f"afftdn=nr={AFFTDN_NR}:nt=w:tn=1",
         "-c:a", "pcm_f32le", str(wav_sortie)],
        "débruitage afftdn")


def executer_moteur(moteur, wav_entree, wav_sortie, args):
    if moteur == "dfn":
        debruiter_dfn(wav_entree, wav_sortie, args.reduction, args.cpu)
    elif moteur == "mossformer2":
        debruiter_mossformer2(wav_entree, wav_sortie)
    else:
        debruiter_afftdn(wav_entree, wav_sortie)


def debruiter(wav_conditionne, wav_decode, travail, args, rapport):
    """Passe 3 avec arbitrage par le QC : en mode auto, DFN d'abord (le
    toucher le plus léger), et si DNSMOS révèle une voix abîmée, essai du
    moteur de secours — le meilleur des deux est retenu, scores à l'appui
    dans le rapport. Constat de terrain : le meilleur moteur dépend de
    l'enregistrement (DFN excellent sur certaines voix, nocif sur d'autres
    quel que soit le réglage de réduction)."""
    import soundfile as sf

    if args.moteur != "auto":
        wav = travail / f"03_debruite_{args.moteur}.wav"
        executer_moteur(args.moteur, wav_conditionne, wav, args)
        rapport["mesures"]["moteur"] = args.moteur
        print(f"  [3/7] débruitage : {args.moteur} (imposé)")
        return wav

    wav_dfn = travail / "03_debruite_dfn.wav"
    executer_moteur("dfn", wav_conditionne, wav_dfn, args)

    try:
        import speechmos  # noqa: F401
    except ImportError:
        rapport["mesures"]["moteur"] = "dfn"
        print("  [3/7] débruitage : dfn (speechmos absent — pas d'arbitrage auto)")
        return wav_dfn

    reference, sr = sf.read(str(wav_decode), dtype="float32")
    if reference.ndim > 1:
        reference = reference.mean(axis=1)
    positions = choisir_fenetres_parole(reference, sr)
    s_ref = score_dnsmos(reference, sr, positions)

    def noter(wav):
        a, _ = sf.read(str(wav), dtype="float32")
        if a.ndim > 1:
            a = a.mean(axis=1)
        return score_dnsmos(a, sr, positions)

    s_dfn = noter(wav_dfn)
    rapport["arbitrage"] = {"reference": s_ref, "dfn": s_dfn}

    if s_ref["sig"] - s_dfn["sig"] <= QC_SIG_BAISSE_MAX:
        rapport["mesures"]["moteur"] = "dfn"
        print(f"  [3/7] débruitage : dfn retenu (voix SIG {s_ref['sig']:.2f} → "
              f"{s_dfn['sig']:.2f}, bruit BAK {s_ref['bak']:.2f} → {s_dfn['bak']:.2f})")
        return wav_dfn

    secours = "mossformer2" if CLEARVOICE_PYTHON.exists() else "afftdn"
    print(f"  [3/7] débruitage : dfn abîme cette voix (SIG {s_ref['sig']:.2f} → "
          f"{s_dfn['sig']:.2f}) — essai {secours}…")
    wav_secours = travail / f"03_debruite_{secours}.wav"
    executer_moteur(secours, wav_conditionne, wav_secours, args)
    s_secours = noter(wav_secours)
    rapport["arbitrage"][secours] = s_secours

    # priorité à la voix : le secours gagne s'il repasse sous le seuil, et
    # même à scores comparables (±0.05) — on n'arrive ici que parce que DFN
    # a déjà échoué au seuil, et ses artefacts (voix métallique/sous l'eau)
    # sont perceptuellement pires que ce que DNSMOS mesure (validé à
    # l'écoute sur enregistrement réel) ; DFN n'est repris que s'il est
    # nettement meilleur
    secours_ok = s_ref["sig"] - s_secours["sig"] <= QC_SIG_BAISSE_MAX
    if secours_ok or s_secours["sig"] >= s_dfn["sig"] - 0.05:
        retenu, wav, s = secours, wav_secours, s_secours
    else:
        retenu, wav, s = "dfn", wav_dfn, s_dfn
    rapport["mesures"]["moteur"] = retenu
    if retenu == "mossformer2":
        rapport["alertes"].append(
            "mossformer2 retenu — ce moteur comprime quelque peu la dynamique "
            "(crêtes réduites ~8 dB, gain non strictement constant, mesuré sur "
            "matériel réel) ; si l'authenticité prime, comparer à l'écoute avec "
            "--moteur afftdn ou le circuit --exporter-rx")
    print(f"        → {retenu} retenu (voix SIG {s_ref['sig']:.2f} → {s['sig']:.2f}, "
          f"bruit BAK {s_ref['bak']:.2f} → {s['bak']:.2f})")
    return wav


# ─────────────────────────────────────────────────────────────────────────────
# Passe 4 — Nivelage doux (optionnel)
# ─────────────────────────────────────────────────────────────────────────────

def niveler(wav_entree, wav_sortie):
    """dynaudnorm en réglages conservateurs.

    Fenêtre effective ~15.5 s (f=500 ms × g=31) : le gain évolue trop
    lentement pour pomper ou pour manger une fin de phrase à l'échelle
    d'une syllabe. m=8 borne l'amplification, t=0.02 interdit d'amplifier
    les trames sous le seuil (silences, fond de salle), b=1 évite
    l'anomalie de gain en début/fin de fichier.
    """
    executer(
        ["ffmpeg", "-y", "-hide_banner", "-i", str(wav_entree),
         "-af", DYNAUDNORM_DOUX, "-c:a", "pcm_f32le", str(wav_sortie)],
        "nivelage")


# ─────────────────────────────────────────────────────────────────────────────
# Passe 5 — Normalisation linéaire contrainte par la crête vraie
# ─────────────────────────────────────────────────────────────────────────────

def mesurer_crete_vraie(audio, sr):
    """Crête vraie (dBTP) par suréchantillonnage 4x, par morceaux (RAM)."""
    from scipy.signal import resample_poly
    crete = 0.0
    pas = 8 * sr  # 8 s par morceau, avec chevauchement pour les bords
    marge = 256
    for canal in audio if audio.ndim > 1 else [audio]:
        for i in range(0, len(canal), pas):
            morceau = canal[max(0, i - marge):i + pas + marge]
            sur = resample_poly(morceau.astype(np.float64), 4, 1)
            crete = max(crete, float(np.abs(sur).max()))
    return 20 * math.log10(max(crete, 1e-12))


def normaliser(wav_entree, wav_sortie, lufs_cible, rapport):
    """Mesure BS.1770 puis gain CONSTANT — la seule normalisation vraiment
    transparente. loudnorm de ffmpeg retombe silencieusement en mode
    dynamique (AGC qui pompe) dès que la cible est inatteignable ; on
    applique donc le gain nous-mêmes et on gère la contrainte de crête
    explicitement :
      - dépassement ≤ 2 dB  → on réduit le gain (fichier un peu moins fort
        que la cible, zéro traitement dynamique)
      - dépassement > 2 dB  → crêtes exceptionnelles (cloches de méditation) :
        limiteur suréchantillonné 192 kHz sur ces seules crêtes, signalé
        dans le rapport
    """
    import soundfile as sf
    import pyloudnorm as pyln

    audio, sr = sf.read(str(wav_entree), dtype="float32")
    metre = pyln.Meter(sr)
    lufs = metre.integrated_loudness(audio)
    tp = mesurer_crete_vraie(audio, sr)

    if lufs < -55:
        raise RuntimeError(f"Fichier quasi muet ({lufs:.1f} LUFS) — vérifier la source")

    gain = lufs_cible - lufs
    if gain > GAIN_MAX_DB:
        rapport["alertes"].append(
            f"gain demandé {gain:.1f} dB > garde-fou {GAIN_MAX_DB} dB — plafonné")
        gain = GAIN_MAX_DB

    tp_apres = tp + gain
    limiteur = False
    if tp_apres > TP_PLAFOND_DB:
        exces = tp_apres - TP_PLAFOND_DB
        if exces <= EXCES_GAIN_REDUIT_DB:
            gain -= exces
            rapport["alertes"].append(
                f"cible réduite de {exces:.1f} dB pour respecter {TP_PLAFOND_DB} dBTP "
                f"sans limiteur (sortie ≈ {lufs + gain:.1f} LUFS)")
        else:
            limiteur = True
            rapport["alertes"].append(
                f"crêtes isolées {exces:.1f} dB au-dessus du plafond (cloches ?) — "
                f"limiteur suréchantillonné engagé sur ces crêtes uniquement")

    rapport["mesures"]["lufs_avant_gain"] = round(float(lufs), 2)
    rapport["mesures"]["tp_avant_gain"] = round(tp, 2)
    rapport["mesures"]["gain_db"] = round(gain, 2)
    rapport["mesures"]["limiteur"] = limiteur

    audio = (audio.astype(np.float64) * db_vers_lineaire(gain)).astype(np.float32)

    if not limiteur:
        sf.write(str(wav_sortie), audio, sr, subtype="FLOAT")
        return

    # Limiteur en dernier recours : suréchantillonnage 192 kHz pour un vrai
    # comportement true-peak, level=disabled (sinon alimiter re-normalise à
    # 0 dB !), latency=1 pour compenser le retard d'anticipation.
    wav_gain = str(wav_sortie) + ".gain.wav"
    sf.write(wav_gain, audio, sr, subtype="FLOAT")
    chaine = (f"aresample=192000,"
              f"alimiter=limit={db_vers_lineaire(TP_PLAFOND_DB):.6f}"
              f":attack=5:release=100:level=disabled:latency=1,"
              f"aresample=resampler=soxr:precision=28:out_sample_rate={sr}")
    executer(["ffmpeg", "-y", "-hide_banner", "-i", wav_gain,
              "-af", chaine, "-c:a", "pcm_f32le", str(wav_sortie)], "limiteur")
    os.remove(wav_gain)


# ─────────────────────────────────────────────────────────────────────────────
# Passe 6 — Encodage MP3
# ─────────────────────────────────────────────────────────────────────────────

def encoder_mp3(wav_entree, source, mp3_sortie, canaux):
    """LAME V0 (jamais battu par le 320 CBR en ABX), tags copiés de la
    source, ID3v2.3 + ID3v1 pour la compatibilité lecteurs anciens."""
    executer(
        ["ffmpeg", "-y", "-hide_banner",
         "-i", str(wav_entree), "-i", str(source),
         "-map", "0:a", "-map_metadata", "1",
         "-ac", str(canaux), "-ar", str(SR_TRAVAIL),
         "-c:a", "libmp3lame", "-q:a", str(MP3_QUALITE_VBR),
         "-id3v2_version", "3", "-write_id3v1", "1",
         str(mp3_sortie)],
        "encodage MP3")


# ─────────────────────────────────────────────────────────────────────────────
# Passe 7 — Contrôle qualité
# ─────────────────────────────────────────────────────────────────────────────

def choisir_fenetres_parole(audio, sr):
    """Choisit des fenêtres actives (RMS au-dessus de la médiane) réparties
    sur toute la durée — on évalue la parole, pas les silences."""
    taille = int(QC_FENETRE_S * sr)
    if len(audio) < taille * 2:
        return [0]
    pas = taille // 2
    positions = np.arange(0, len(audio) - taille, pas)
    rms = np.array([float(np.sqrt(np.mean(audio[p:p + taille] ** 2))) for p in positions])
    actives = positions[rms > np.median(rms)]
    if len(actives) == 0:
        actives = positions
    indices = np.linspace(0, len(actives) - 1, min(QC_FENETRES, len(actives))).astype(int)
    return [int(actives[i]) for i in indices]


def score_dnsmos(audio, sr, positions):
    """DNSMOS P.835 moyen sur les fenêtres données (SIG/BAK/OVRL, échelle 1-5).

    Chaque fenêtre est ramenée au même RMS (-26 dBFS) pour que la
    comparaison avant/après mesure la qualité, pas la différence de gain.
    """
    import librosa
    from speechmos import dnsmos

    taille = int(QC_FENETRE_S * sr)
    scores = {"sig": [], "bak": [], "ovrl": []}
    for p in positions:
        fenetre = audio[p:p + taille]
        rms = float(np.sqrt(np.mean(fenetre ** 2)))
        if rms < 1e-6:
            continue
        fenetre = fenetre * (db_vers_lineaire(-26) / rms)
        fenetre = np.clip(fenetre, -1.0, 1.0)
        fenetre_16k = librosa.resample(fenetre, orig_sr=sr, target_sr=16000)
        r = dnsmos.run(fenetre_16k, sr=16000)
        scores["sig"].append(r["sig_mos"])
        scores["bak"].append(r["bak_mos"])
        scores["ovrl"].append(r["ovrl_mos"])
    if not scores["sig"]:
        return None
    return {k: round(float(np.mean(v)), 3) for k, v in scores.items()}


def controler_qualite(wav_origine, wav_final, mp3_final, rapport):
    import soundfile as sf

    # DNSMOS avant/après aux mêmes positions (le pipeline préserve le temps)
    try:
        import speechmos  # noqa: F401
        avant, sr = sf.read(str(wav_origine), dtype="float32")
        if avant.ndim > 1:
            avant = avant.mean(axis=1)
        apres, _ = sf.read(str(wav_final), dtype="float32")
        if apres.ndim > 1:
            apres = apres.mean(axis=1)
        positions = choisir_fenetres_parole(avant, sr)
        s_avant = score_dnsmos(avant, sr, positions)
        s_apres = score_dnsmos(apres, sr, positions)
        if s_avant and s_apres:
            rapport["dnsmos"] = {"avant": s_avant, "apres": s_apres}
            if s_apres["sig"] < s_avant["sig"] - QC_SIG_BAISSE_MAX:
                rapport["alertes"].append(
                    f"DNSMOS : la qualité vocale a baissé (SIG {s_avant['sig']} → "
                    f"{s_apres['sig']}) — ÉCOUTE MANUELLE RECOMMANDÉE, essayer un "
                    f"autre --moteur ou le circuit --exporter-rx")
            if s_apres["bak"] <= s_avant["bak"]:
                rapport["alertes"].append(
                    f"DNSMOS : le bruit de fond n'a pas diminué (BAK "
                    f"{s_avant['bak']} → {s_apres['bak']})")
    except ImportError:
        rapport["dnsmos"] = "speechmos absent — non mesuré"

    # Vérification indépendante du MP3 final (ebur128, crête vraie 4x)
    stderr = executer(
        ["ffmpeg", "-nostats", "-hide_banner", "-i", str(mp3_final),
         "-filter_complex", "ebur128=peak=true", "-f", "null", "-"],
        "vérification ebur128")
    bloc = stderr[stderr.rfind("Summary:"):]
    m_i = re.search(r"I:\s+(-?[\d.]+) LUFS", bloc)
    m_tp = re.search(r"Peak:\s+(-?[\d.]+) dBFS", bloc)
    if m_i:
        rapport["mesures"]["lufs_mp3_final"] = float(m_i.group(1))
    if m_tp:
        tp_final = float(m_tp.group(1))
        rapport["mesures"]["tp_mp3_final"] = tp_final
        if tp_final > -0.5:
            rapport["alertes"].append(
                f"crête vraie du MP3 final {tp_final:.2f} dBTP > -0.5 dBTP "
                f"(dépassement d'encodage anormal)")

    # Durées identiques (aucune passe ne doit décaler ou tronquer le temps)
    d_avant = analyser_source(wav_origine)["duree"]
    d_apres = analyser_source(mp3_final)["duree"]
    rapport["mesures"]["duree_s"] = round(d_apres, 2)
    if abs(d_avant - d_apres) > 0.15:
        rapport["alertes"].append(
            f"durée modifiée : {d_avant:.2f} s → {d_apres:.2f} s")


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────

def traiter_fichier(fichier, args):
    debut = time.time()
    fichier = Path(fichier)
    dossier_sortie = Path(args.sortie) if args.sortie else fichier.parent
    dossier_sortie.mkdir(parents=True, exist_ok=True)
    mp3_sortie = dossier_sortie / f"{fichier.stem}_nettoye.mp3"

    if mp3_sortie.exists() and not args.forcer:
        print(f"  ↷ {mp3_sortie.name} existe déjà — ignoré (--forcer pour refaire)")
        return None

    canaux = 2 if args.stereo else 1
    lufs_cible = args.lufs if args.lufs is not None else (
        LUFS_CIBLE_STEREO if args.stereo else LUFS_CIBLE_MONO)

    rapport = {
        "source": str(fichier), "sortie": str(mp3_sortie),
        "parametres": {"moteur": args.moteur, "reduction_db": args.reduction,
                       "lufs_cible": lufs_cible, "tp_plafond": TP_PLAFOND_DB,
                       "niveler": args.niveler, "canaux": canaux},
        "mesures": {}, "alertes": [],
    }

    travail = Path(tempfile.mkdtemp(prefix=f"nettoyer_{fichier.stem[:30]}_"))
    try:
        # Passe 1 — analyse
        infos = analyser_source(fichier)
        rapport["mesures"]["source"] = infos
        print(f"  [1/7] analyse : {infos['duree']/60:.1f} min, {infos['sr']} Hz, "
              f"{infos['canaux']} canaux")

        # Passe 2 — conditionnement
        wav_decode = travail / "01_decode.wav"
        decoder(fichier, wav_decode, canaux)
        notchs = detecter_ronflette(wav_decode)
        rapport["mesures"]["ronflette_hz"] = notchs
        wav_conditionne = travail / "02_conditionne.wav"
        conditionner(wav_decode, wav_conditionne, notchs, args.passe_haut)
        print(f"  [2/7] conditionnement : passe-haut {args.passe_haut} Hz"
              + (f", notchs {notchs}" if notchs else ", pas de ronflette détectée"))

        if args.exporter_rx:
            wav_rx = dossier_sortie / f"{fichier.stem}_pour_rx.wav"
            shutil.copy(wav_conditionne, wav_rx)
            print(f"  → exporté pour iZotope RX : {wav_rx}\n"
                  f"    (débruiter dans RX puis relancer avec --importer-rx)")
            return None

        # Passe 3 — débruitage (avec arbitrage QC en mode auto)
        if args.importer_rx:
            wav_debruite = travail / "03_debruite.wav"
            decoder(args.importer_rx, wav_debruite, canaux)
            print(f"  [3/7] débruitage : importé depuis {args.importer_rx}")
        else:
            wav_debruite = debruiter(wav_conditionne, wav_decode, travail, args, rapport)

        # Passe 4 — nivelage (optionnel)
        if args.niveler:
            wav_nivele = travail / "04_nivele.wav"
            niveler(wav_debruite, wav_nivele)
            print("  [4/7] nivelage doux appliqué (fenêtre ~15 s)")
        else:
            wav_nivele = wav_debruite
            print("  [4/7] nivelage : désactivé (défaut — dynamique d'origine préservée)")

        # Passe 5 — normalisation linéaire
        wav_normalise = travail / "05_normalise.wav"
        normaliser(wav_nivele, wav_normalise, lufs_cible, rapport)
        m = rapport["mesures"]
        print(f"  [5/7] normalisation : {m['lufs_avant_gain']} LUFS "
              f"→ gain {m['gain_db']:+.1f} dB"
              + (" + limiteur crêtes" if m["limiteur"] else " (gain pur)"))

        # Passe 6 — encodage
        encoder_mp3(wav_normalise, fichier, mp3_sortie, canaux)
        print(f"  [6/7] encodage : MP3 LAME V0 {'stéréo' if canaux == 2 else 'mono'}")

        # Passe 7 — contrôle qualité
        controler_qualite(wav_decode, wav_normalise, mp3_sortie, rapport)
        if "dnsmos" in rapport and isinstance(rapport["dnsmos"], dict):
            av, ap = rapport["dnsmos"]["avant"], rapport["dnsmos"]["apres"]
            print(f"  [7/7] QC DNSMOS — voix (SIG) : {av['sig']} → {ap['sig']}, "
                  f"bruit (BAK) : {av['bak']} → {ap['bak']}, "
                  f"global : {av['ovrl']} → {ap['ovrl']}")
        print(f"        LUFS final : {rapport['mesures'].get('lufs_mp3_final', '?')}, "
              f"crête vraie : {rapport['mesures'].get('tp_mp3_final', '?')} dBTP")

        rapport["duree_traitement_s"] = round(time.time() - debut, 1)
        chemin_rapport = dossier_sortie / f"{fichier.stem}_nettoyage.json"
        chemin_rapport.write_text(json.dumps(rapport, indent=2, ensure_ascii=False))

        for alerte in rapport["alertes"]:
            print(f"  ⚠ {alerte}")
        print(f"  ✓ {mp3_sortie.name} ({rapport['duree_traitement_s']:.0f} s)")
        return rapport
    finally:
        if args.garder_travail:
            print(f"  (fichiers intermédiaires conservés : {travail})")
        else:
            shutil.rmtree(travail, ignore_errors=True)


def principal():
    parseur = argparse.ArgumentParser(
        description="Restauration audio à toucher léger pour causeries du Dhamma "
                    "(débruitage DeepFilterNet3 + normalisation linéaire BS.1770)",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parseur.add_argument("entree", help="fichier audio ou dossier à traiter")
    parseur.add_argument("-o", "--sortie", help="dossier de sortie (défaut : à côté de la source)")
    parseur.add_argument("--moteur", choices=["auto", "dfn", "mossformer2", "afftdn"],
                         default="auto",
                         help="moteur de débruitage : auto (défaut — DFN, puis bascule "
                              "arbitrée par DNSMOS si la voix en souffre), dfn "
                              "(DeepFilterNet3), mossformer2 (ClearerVoice, env conda "
                              "`clearvoice`), afftdn (spectral classique très doux)")
    parseur.add_argument("--reduction", type=int, default=REDUCTION_DB,
                         help=f"atténuation max du bruit en dB pour le moteur dfn — "
                              f"le mélange dry/wet : 10=très léger, 15=défaut, "
                              f"20=plus net (défaut {REDUCTION_DB})")
    parseur.add_argument("--lufs", type=float, default=None,
                         help=f"cible de sonie (défaut {LUFS_CIBLE_MONO} mono, "
                              f"{LUFS_CIBLE_STEREO} stéréo)")
    parseur.add_argument("--passe-haut", type=int, default=PASSE_HAUT_HZ, metavar="HZ",
                         help=f"fréquence du passe-haut en Hz, 0 pour désactiver "
                              f"(défaut {PASSE_HAUT_HZ})")
    parseur.add_argument("--niveler", action="store_true",
                         help="nivelage dynamique doux intra-fichier (défaut : désactivé)")
    parseur.add_argument("--stereo", action="store_true",
                         help="conserver la stéréo (défaut : repli mono — les causeries "
                              "sont mono, le repli moyenne aussi le bruit)")
    parseur.add_argument("--cpu", action="store_true", help="forcer le débruitage sur CPU")
    parseur.add_argument("--exporter-rx", action="store_true",
                         help="écrire le WAV conditionné pour débruitage manuel iZotope RX, "
                              "puis s'arrêter")
    parseur.add_argument("--importer-rx", metavar="FICHIER",
                         help="reprendre après débruitage manuel RX avec ce WAV")
    parseur.add_argument("--forcer", action="store_true",
                         help="retraiter même si la sortie existe")
    parseur.add_argument("--garder-travail", action="store_true",
                         help="conserver les WAV intermédiaires (débogage/écoute A-B)")
    args = parseur.parse_args()

    verifier_dependances(args)

    entree = Path(args.entree)
    if entree.is_dir():
        fichiers = sorted(f for f in entree.iterdir()
                          if f.suffix.lower() in EXTENSIONS
                          and not f.stem.endswith("_nettoye")
                          and not f.stem.endswith("_pour_rx"))
        if args.importer_rx:
            print("--importer-rx ne s'applique qu'à un fichier unique")
            sys.exit(1)
    else:
        fichiers = [entree]
    if not fichiers:
        print(f"Aucun fichier audio dans {entree}")
        sys.exit(1)

    print(f"─── nettoyer.py — {len(fichiers)} fichier(s) ───")
    echecs = []
    for i, f in enumerate(fichiers, 1):
        print(f"[{i}/{len(fichiers)}] {f.name}")
        try:
            traiter_fichier(f, args)
        except Exception as e:
            echecs.append((f.name, str(e)))
            print(f"  ✗ échec : {e}")

    fermer_pont()
    if echecs:
        print(f"\n{len(echecs)} échec(s) :")
        for nom, err in echecs:
            print(f"  - {nom} : {err.splitlines()[0] if err else err}")
        sys.exit(1)


if __name__ == "__main__":
    principal()
