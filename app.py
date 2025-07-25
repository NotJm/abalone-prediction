from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import numpy as np
import logging

app = Flask(__name__)

# Configurar el registro
logging.basicConfig(level=logging.DEBUG)

# Cargar los modelos entrenados
try:
    # Cargar el modelo de red neuronal
    model = joblib.load('network_neuronal.pkl')
    app.logger.debug('Modelo de red neuronal cargado correctamente.')
    
    # Cargar el scaler
    scaler = joblib.load('scaler_network_neuronal.pkl')
    app.logger.debug('Scaler cargado correctamente.')
    
except Exception as e:
    app.logger.error(f'Error al cargar los modelos: {str(e)}')
    model = None
    scaler = None

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Verificar que los modelos estén cargados
        if model is None or scaler is None:
            return jsonify({'error': 'Los modelos no están disponibles'}), 500
        
        # Obtener los datos enviados en el request
        rings = float(request.form['rings'])
        whole_wt = float(request.form['whole_wt'])
        shell_wt = float(request.form['shell_wt'])
        shucked_wt = float(request.form['shucked_wt'])
        diameter = float(request.form['diameter'])

        expected_columns: list[str] = [
            'length', 'diameter', 'height', 'whole_wt', 'shucked_wt', 
            'viscera_wt', 'shell_wt', 'rings', 'sex_F', 'sex_I', 'sex_M'
        ]
        
        features: list[str] = ['rings', 'whole_wt', 'shell_wt', 'shucked_wt', 'diameter']

        data_dict: dict = {
            'length': 0,           
            'diameter': diameter,
            'height': 0,          
            'whole_wt': whole_wt,
            'shucked_wt': shucked_wt,
            'viscera_wt': 0,       
            'shell_wt': shell_wt,
            'rings': rings,
            'sex_F': 0,            
            'sex_I': 0,
            'sex_M': 0             
        }

        data_df = pd.DataFrame([data_dict], columns=expected_columns)
        
        app.logger.debug(f'DataFrame creado: {data_df}')
        
        # Escalar los datos usando el scaler entrenado
        data_scaled = scaler.transform(data_df)
        app.logger.debug(f'Datos escalados: {data_scaled}')
        
        data_scaled_df = pd.DataFrame(data_scaled, columns=expected_columns)
        
        X = data_scaled_df[features]
        
        app.logger.debug(f'Datos seleccionados:{X}')

        # Realizar predicción con el modelo de red neuronal
        prediction = model.predict(X)
        
        # La predicción puede ser un array, tomar el primer valor
        edad_estimada = float(prediction[0])
        
        edad_estimada = round(edad_estimada, 1)
                
        app.logger.debug(f'Predicción de edad: {edad_estimada}')

        # Devolver la predicción como respuesta JSON
        return jsonify({'edad_estimada': edad_estimada})
        
    except ValueError as ve:
        app.logger.error(f'Error de valor en la predicción: {str(ve)}')
        return jsonify({'error': 'Valores de entrada inválidos. Asegúrate de ingresar números válidos.'}), 400
    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return jsonify({'error': f'Error interno del servidor: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint para verificar el estado de la aplicación y los modelos"""
    try:
        model_status = "OK" if model is not None else "ERROR"
        scaler_status = "OK" if scaler is not None else "ERROR"
        
        return jsonify({
            'status': 'OK' if model_status == 'OK' and scaler_status == 'OK' else 'ERROR',
            'model_status': model_status,
            'scaler_status': scaler_status
        })
    except Exception as e:
        return jsonify({'status': 'ERROR', 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
