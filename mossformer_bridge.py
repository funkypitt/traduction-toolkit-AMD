#!/usr/bin/env python3
"""
Pont MossFormer2 (ClearerVoice-Studio) pour nettoyer.py
==================================================================================
Tourne sous ~/miniconda3/envs/clearvoice/bin/python (dépendances incompatibles
avec l'env principal `interview`). Protocole JSON-lines sur stdin/stdout,
stdout protégé (les bibliothèques bavardes sont redirigées vers stderr) —
même patron que les ponts TTS du dépôt.

Commandes :
  {"cmd": "init"}                                → charge MossFormer2_SE_48K
  {"cmd": "traiter", "entree": "...", "sortie": "..."} → débruite un WAV 48 kHz
  {"cmd": "quit"}

Création de l'environnement :
  conda create -n clearvoice python=3.10 -y
  ~/miniconda3/envs/clearvoice/bin/python -m pip install clearvoice
"""

import json
import os
import sys

# Canal JSON protégé : on duplique stdout puis on redirige le descripteur 1
# vers stderr — tout print() de bibliothèque part sur stderr, le JSON reste pur.
canal_json = os.fdopen(os.dup(1), "w")
os.dup2(2, 1)

cv = None


def repondre(objet):
    canal_json.write(json.dumps(objet) + "\n")
    canal_json.flush()


for ligne in sys.stdin:
    ligne = ligne.strip()
    if not ligne:
        continue
    try:
        requete = json.loads(ligne)
        cmd = requete.get("cmd")

        if cmd == "init":
            from clearvoice import ClearVoice
            cv = ClearVoice(task="speech_enhancement",
                            model_names=["MossFormer2_SE_48K"])
            repondre({"ok": True})

        elif cmd == "traiter":
            import numpy as np
            import soundfile as sf
            resultat = cv(input_path=requete["entree"], online_write=False)
            resultat = np.asarray(resultat).squeeze()
            sf.write(requete["sortie"], resultat.astype(np.float32),
                     48000, subtype="FLOAT")
            repondre({"ok": True, "echantillons": int(len(resultat))})

        elif cmd == "quit":
            repondre({"ok": True})
            break

        else:
            repondre({"ok": False, "erreur": f"commande inconnue : {cmd}"})

    except Exception as e:
        repondre({"ok": False, "erreur": f"{type(e).__name__}: {e}"})
