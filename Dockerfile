# Copie le fichier config.json dans l'image
COPY config.json ./config.json

Commande par défaut à exécuter
CMD ["node", "app.js"]
