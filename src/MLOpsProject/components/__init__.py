import sys
import os

# ✅ Ajoute le dossier 'src' au chemin d'import pour permettre les imports depuis ce dossier
chemin_src = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'src'))
if chemin_src not in sys.path:
    sys.path.insert(0, chemin_src)
