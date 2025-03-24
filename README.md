# Pokémon Battle Predictor 🚀

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A machine learning-powered web app that predicts the outcome of Pokémon battles based on stats and type matchups, with beautiful visualizations.

![App Screenshot](https://res.cloudinary.com/dyl5ibyvg/image/upload/v1742846860/hclp1f4qwj79ymecsl95.png)


![App Screenshot](https://res.cloudinary.com/dyl5ibyvg/image/upload/v1742846860/vayroztnvggom4wkpq6n.png)

## 📺 Video Demo
[![YouTube Demo](https://img.shields.io/badge/YouTube-Demo-red)](https://youtu.be/your-demo-link-here)

*(Click the badge above to watch the demo video)*

## ✨ Features

- 🧠 ML-powered battle predictions (85%+ accuracy)
- ⚡ Real-time type advantage calculations
- 📊 Interactive stat radar charts
- 🎨 Animated battle analysis visuals
- 📱 Fully responsive design
- 🗃️ Supports all 898 Pokémon

## 🛠️ Tech Stack

- **Backend**: Python, Flask
- **Frontend**: HTML5, CSS3, JavaScript, Chart.js
- **Machine Learning**: Scikit-learn, XGBoost
- **Data**: Pokémon stats dataset (CSV)

## 🚀 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/pokemon-battle-predictor.git
   cd pokemon-battle-predictor
   ```

2. Create a virtual environment:

   ```python
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
4. Download Pokémon dataset:

   ```bash
   wget https://raw.githubusercontent.com/veekun/pokedex/master/pokedex/data/csv/pokemon.csv -O data/pokemon.csv
   ```

## 🏃‍♂️ Running the App

1. Train the ML model (first time only):

   ```bash
   python advanced_predictor.py
   ```
2. Start the Flask server:

   ```bash
   python app.py
   ```
3. Open in browser:

   ``` 
   http://localhost:5000
   ```

## 🎮 How to Use

1. Select two Pokémon from the dropdown
2. Click "Predict Winner"
3. View:
   - Winner prediction with confidence %
   - Type advantage visualization
   - Speed comparison
   - Interactive stat radar chart

## 📂 Project Structure


pokemon-battle-predictor/
- ├── data/                  # Pokémon datasets
- ├── models/                # Trained ML models
- ├── static/                # Static files
- │   ├── css/               # Stylesheets
- │   ├── icons/             # App icons
- │   └── js/                # JavaScript files
- ├── templates/             # Flask templates
- ├── app.py                 # Main application
- ├── train_model.py         # Model training script
- ├── requirements.txt       # Dependencies
- └── README.md              # This file


## 📝 Customizing

### Add New Pokémon

1. Edit `data/pokemon.csv` following the existing format
2. Retrain the model:
   ```bash
   python train_model.py
   ```

### Change Visual Style

Edit the CSS variables in `static/css/main.css`:

```css
:root {
  --primary-color: #ff3e50;
  --secondary-color: #00b4d8;
}
```

## 🤝 Contributing

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

## ✉️ Contact

Your Name - rjaahmad60@gmail.com

Project Link: [https://github.com/Ahmad10Raza/pokemon-battle-predictor](https://github.com/Ahmad10Raza/pokemon-battle-predictor)



