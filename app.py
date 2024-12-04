from flask import Flask, render_template, request
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, recall_score, f1_score 
import numpy as np


# Créez l'application Flask
app = Flask(__name__)

# Chemin vers les données et le modèle
data_path = 'data/train.csv'
model_path = 'models/trained_model.pkl'

# Vérifier si le fichier de données existe
if not os.path.exists(data_path):
    print("Le fichier data/train.csv est introuvable.")
    exit(1)

# Charger les données
data = pd.read_csv(data_path)
X = data.drop('price_range', axis=1)
y = data['price_range']

# Charger le modèle ou entraîner un nouveau modèle si nécessaire
model = None
if os.path.exists(model_path):
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
else:
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    # Sauvegarder le modèle
    os.makedirs('models', exist_ok=True)
    with open(model_path, 'wb') as model_file:
        pickle.dump(model, model_file)
    print(f"Modèle entraîné et sauvegardé dans {model_path}")

# Route principale (index)
@app.route('/')
def index():
    return render_template('index.html')

# Route pour explorer les données
@app.route('/explore_data')
def explore_data():
    # Vérifiez si le fichier de données existe
    if not os.path.exists(data_path):
        return "Le fichier data/train.csv est introuvable.", 404

    # Charger les données
    data = pd.read_csv(data_path)

    # Afficher les premières lignes des données sous forme de tableau HTML
    head_data = data.head().to_html(classes='table table-striped')

    # Passer la table complète au template
    full_data = data.head().to_html(classes='table table-striped') 

    # Statistiques descriptives
    descriptive_stats = data.describe().to_html(classes='table table-striped')

    return render_template('explore_data.html', head_data=head_data, full_data=full_data, descriptive_stats=descriptive_stats)

# Route pour visualiser les données
@app.route('/visualize_data')
def visualize_data():
    # Vérifiez si le fichier de données existe
    if not os.path.exists(data_path):
        return "Le fichier data/train.csv est introuvable.", 404

    # Charger les données
    data = pd.read_csv(data_path)

    # Vérifier et créer le dossier pour les visualisations
    output_dir = 'static/visualizations'
    os.makedirs(output_dir, exist_ok=True)

    # Liste des graphiques à créer
    image_urls = []

    # 1. Battery Power vs Price Range
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='price_range', y='battery_power', data=data)
    plt.title("Battery Power vs Price Range")
    battery_power_image_path = os.path.join(output_dir, 'battery_power_vs_price.png')
    plt.savefig(battery_power_image_path)
    plt.close()
    image_urls.append('/' + battery_power_image_path)

    # 2. RAM vs Price Range
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='price_range', y='ram', data=data)
    plt.title("RAM vs Price Range")
    ram_image_path = os.path.join(output_dir, 'ram_vs_price.png')
    plt.savefig(ram_image_path)
    plt.close()
    image_urls.append('/' + ram_image_path)

    # 3. Mobile Weight vs Price Range
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='price_range', y='mobile_wt', data=data)
    plt.title("Mobile Weight vs Price Range")
    mobile_wt_image_path = os.path.join(output_dir, 'mobile_wt_vs_price.png')
    plt.savefig(mobile_wt_image_path)
    plt.close()
    image_urls.append('/' + mobile_wt_image_path)

    # 4. RAM vs Battery Power
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='ram', y='battery_power', data=data)
    plt.title("RAM vs Battery Power")
    ram_battery_image_path = os.path.join(output_dir, 'ram_vs_battery.png')
    plt.savefig(ram_battery_image_path)
    plt.close()
    image_urls.append('/' + ram_battery_image_path)

    # 5. Screen Height vs Price Range
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='price_range', y='sc_h', data=data)
    plt.title("Screen Height vs Price Range")
    screen_height_image_path = os.path.join(output_dir, 'screen_height_vs_price.png')
    plt.savefig(screen_height_image_path)
    plt.close()
    image_urls.append('/' + screen_height_image_path)

    # 6. Mobile Weight vs Screen Width
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='mobile_wt', y='sc_w', data=data)
    plt.title("Mobile Weight vs Screen Width")
    mobile_weight_screen_width_image_path = os.path.join(output_dir, 'mobile_weight_vs_screen_width.png')
    plt.savefig(mobile_weight_screen_width_image_path)
    plt.close()
    image_urls.append('/' + mobile_weight_screen_width_image_path)

    # 7. Battery Power vs Screen Height
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='battery_power', y='sc_h', data=data)
    plt.title("Battery Power vs Screen Height")
    battery_power_screen_height_image_path = os.path.join(output_dir, 'battery_power_vs_screen_height.png')
    plt.savefig(battery_power_screen_height_image_path)
    plt.close()
    image_urls.append('/' + battery_power_screen_height_image_path)

    # 8. Camera Resolution (pc) vs Price Range
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='price_range', y='pc', data=data)
    plt.title("Camera Resolution vs Price Range")
    camera_resolution_image_path = os.path.join(output_dir, 'camera_resolution_vs_price.png')
    plt.savefig(camera_resolution_image_path)
    plt.close()
    image_urls.append('/' + camera_resolution_image_path)
    # Rendre le modèle HTML avec la liste des images générées
    return render_template('visualize_data.html', image_urls=image_urls)

# Route pour entraîner le modèle
@app.route('/train_model')
def train_model():
    # Vérifiez si le fichier de données existe
    if not os.path.exists(data_path):
        return "Le fichier data/train.csv est introuvable.", 404

    # Charger les données
    try:
        data = pd.read_csv(data_path)
    except Exception as e:
        return f"Erreur lors du chargement des données : {e}", 500
    
    # Vérifier que la colonne 'price_range' existe dans les données
    if 'price_range' not in data.columns:
        return "La colonne 'price_range' est manquante dans les données.", 400
    
    X = data.drop('price_range', axis=1)
    y = data['price_range']

    # Entraîner le modèle (RandomForestClassifier)
    trained_model = RandomForestClassifier(n_estimators=100)
    trained_model.fit(X, y)

    # Calculer la précision (accuracy)
    accuracy = trained_model.score(X, y)  # Accuracy sur l'ensemble d'entraînement

    # Prédictions sur l'ensemble d'entraînement pour calculer la matrice de confusion, le rappel et le F1-Score
    y_pred = trained_model.predict(X)

    # Matrice de confusion
    conf_matrix = confusion_matrix(y, y_pred)
    
    # Calcul du recall (rappel) et du F1-score
    recall = recall_score(y, y_pred, average='macro')
    f1 = f1_score(y, y_pred, average='macro')

    # Visualisation de la matrice de confusion avec Seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.title('Matrice de confusion')
    plt.ylabel('Vérité terrain')
    plt.xlabel('Prédictions')
    confusion_image_path = os.path.join('static', 'confusion_matrix.png')
    plt.savefig(confusion_image_path)
    plt.close()

    # Sauvegarder le modèle pour les prédictions futures
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'trained_model.pkl')  # Assurez-vous que le chemin du modèle est correct

    try:
        with open(model_path, 'wb') as model_file:
            pickle.dump(trained_model, model_file)
    except Exception as e:
        return f"Erreur lors de la sauvegarde du modèle : {e}", 500

    # Passer l'accuracy, recall, F1-score et l'image de la matrice de confusion au template
    return render_template('train_model.html', 
                           message="Modèle entraîné et sauvegardé avec succès.",
                           accuracy=accuracy,
                           recall=recall,
                           f1_score=f1,
                           confusion_image='/static/confusion_matrix.png')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    global model
    if request.method == 'POST':
        try:
            # Utilisez la méthode .get() pour chaque champ et définissez une valeur par défaut
            features = [
                float(request.form.get('battery_power', 0)) if request.form.get('battery_power') else 0,  # Valeur par défaut 0 si vide
                int(request.form.get('blue', 0)) if request.form.get('blue') else 0,
                float(request.form.get('clock_speed', 0)) if request.form.get('clock_speed') else 0,
                int(request.form.get('dual_sim', 0)) if request.form.get('dual_sim') else 0,
                float(request.form.get('fc', 0)) if request.form.get('fc') else 0,
                int(request.form.get('four_g', 0)) if request.form.get('four_g') else 0,
                int(request.form.get('int_memory', 0)) if request.form.get('int_memory') else 0,
                float(request.form.get('m_dep', 0)) if request.form.get('m_dep') else 0,
                float(request.form.get('mobile_wt', 0)) if request.form.get('mobile_wt') else 0,
                int(request.form.get('n_cores', 0)) if request.form.get('n_cores') else 0,
                float(request.form.get('pc', 0)) if request.form.get('pc') else 0,
                int(request.form.get('px_height', 0)) if request.form.get('px_height') else 0,
                int(request.form.get('px_width', 0)) if request.form.get('px_width') else 0,
                int(request.form.get('ram', 0)) if request.form.get('ram') else 0,
                int(request.form.get('sc_h', 0)) if request.form.get('sc_h') else 0,
                int(request.form.get('sc_w', 0)) if request.form.get('sc_w') else 0,
                int(request.form.get('talk_time', 0)) if request.form.get('talk_time') else 0,
                int(request.form.get('three_g', 0)) if request.form.get('three_g') else 0,
                int(request.form.get('touch_screen', 0)) if request.form.get('touch_screen') else 0,
                int(request.form.get('wifi', 0)) if request.form.get('wifi') else 0,
            ]

            # Vérifier si le modèle est chargé
            if model is None:
                return "Le modèle n'est pas disponible. Entraînez-le avant de faire des prédictions.", 500

            # Prédire la gamme de prix
            prediction_result = model.predict([features])[0]

            # Retourner le résultat
            return render_template('prediction.html', prediction=prediction_result)

        except Exception as e:
            return f"Erreur lors de la prédiction : {str(e)}", 500

    return render_template('prediction.html')
# Lancer l'application Flask
if __name__ == '__main__':
    app.run(debug=True)
