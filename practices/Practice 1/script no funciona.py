import tkinter as tk
from tkinter import ttk
from queue import Queue
import re
import shutil
import pandas as pd
import soundfile as sf
import pyaudio
import numpy as np
import wave
import speech_recognition as sr
import pyttsx3
import threading
import keyboard
from pydub import AudioSegment
from pydub.utils import mediainfo
import librosa
import noisereduce as nr
from pydub.silence import detect_nonsilent
import random
from scipy.ndimage import shift
from scipy import stats
from scipy.signal import find_peaks
from tqdm.notebook import tqdm


# Librerías de Machine Learning de scikit-learn
from sklearn.model_selection import StratifiedShuffleSplit, cross_validate, train_test_split
from sklearn.metrics import make_scorer, f1_score, recall_score, confusion_matrix, accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.feature_selection import mutual_info_classif, SelectKBest, RFE
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.inspection import permutation_importance
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier
import joblib
import warnings
warnings.filterwarnings("ignore", message="n_fft=.* is too large for input signal of length=.*", category=UserWarning, module="librosa")
def cargar_modelo(ruta_modelo):
    """
    Carga un modelo guardado desde un archivo .pkl

    Parámetros:
    -----------
    ruta_modelo : str
        Ruta completa al archivo .pkl del modelo guardado

    Retorna:
    --------
    model
        El modelo cargado
    """
    try:
        modelo = joblib.load(ruta_modelo)
        print(f"Modelo cargado correctamente desde: {ruta_modelo}")
        return modelo
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo en la ruta {ruta_modelo}")
    except Exception as e:
        print(f"Error al cargar el modelo: {str(e)}")

def extraer_caracteristicas(y_aug, sr_aug):
    # Carga del audio
    y, sr = y_aug, sr_aug
    
    # Dominio del tiempo
    feats = {}
    feats['rms'] = np.sqrt(np.mean(y**2))
    feats['tasa_cruces_cero'] = librosa.feature.zero_crossing_rate(y).mean()
    feats['proporcion_silencio'] = np.mean(librosa.feature.rms(y=y) < 0.01)
    feats['amplitud_maxima'] = np.max(np.abs(y))
    feats['amplitud_minima'] = np.min(np.abs(y))
    feats['amplitud_media'] = np.mean(np.abs(y))
    feats['varianza_amplitud'] = np.var(y)
    feats['asimetria_amplitud'] = stats.skew(y)
    feats['curtosis_amplitud'] = stats.kurtosis(y)
    feats['mediana_amplitud'] = np.median(np.abs(y))
    feats['rango_intercuartilico'] = stats.iqr(y)
    
    # Pico de la autocorrelación
    autocorr = np.correlate(y, y, mode='full')
    peaks, _ = find_peaks(autocorr)
    feats['pico_autocorrelacion'] = np.max(autocorr[peaks]) if peaks.size > 0 else 0

    # Cruces de umbral (umbral = 0)
    feats['cruces_umbral'] = len(np.where(np.diff(y > 0))[0])
    
    # Entropía temporal
    hist, _ = np.histogram(y, bins=20, density=True)
    feats['entropia_temporal'] = stats.entropy(hist)
    
    # Dominio frecuencial
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    feats['centroide_espectral_media'] = np.mean(centroid)
    feats['centroide_espectral_varianza'] = np.var(centroid)
    feats['centroide_espectral_asimetria'] = stats.skew(centroid[0])
    feats['centroide_espectral_curtosis'] = stats.kurtosis(centroid[0])
    
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    feats['ancho_banda_media'] = np.mean(bandwidth)
    feats['ancho_banda_varianza'] = np.var(bandwidth)
    
    feats['rolloff_espectral_media'] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    feats['planitud_espectral_media'] = np.mean(librosa.feature.spectral_flatness(y=y))
    feats['contrast_espectral_media'] = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
    
    
    # MFCC (1-13)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i, mfcc in enumerate(mfccs, start=1):
        feats[f'mfcc{i}_media'] = np.mean(mfcc)
    
    # Características Croma (1-12)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    for i, c in enumerate(chroma, start=1):
        feats[f'chroma{i}_media'] = np.mean(c)
    
    # Tonnetz (1-6)
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    for i, t in enumerate(tonnetz, start=1):
        feats[f'tonnetz{i}_media'] = np.mean(t)
    
    S = np.abs(librosa.stft(y))   
   
    feats['coeficiente_polinomico0'] = np.mean(librosa.feature.poly_features(S=S, order=0))

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    feats['densidad_onset'] = np.mean(onset_env)
    
    return feats








# Configuración global (las funciones cargar_modelo y extraer_caracteristicas deben estar definidas)
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
THRESHOLD = 50

class VoiceRecognitionApp:
    def __init__(self, master):
        self.master = master
        master.title("Selección de Modelo SVM")
        
        # Cargar modelos
        self.models = {
            'Modelo 01 SVM': cargar_modelo('01_svm.pkl'),
            'Modelo 02 SVM': cargar_modelo('02_svm.pkl'),
            'Modelo 03 SVM': cargar_modelo('03_svm.pkl')
        }
        
        # Interfaz gráfica
        self.create_widgets()
        self.stop_event = threading.Event()
        self.log_queue = Queue()
        self.master.after(100, self.update_log)

    def create_widgets(self):
        # Selección de modelo
        self.model_var = tk.StringVar(value='Modelo 01 SVM')
        ttk.Label(self.master, text="Seleccionar modelo:").pack(pady=5)
        self.model_menu = ttk.Combobox(
            self.master, 
            textvariable=self.model_var,
            values=list(self.models.keys()),
            state='readonly'
        )
        self.model_menu.pack(pady=5)
        
        # Botones
        self.btn_frame = ttk.Frame(self.master)
        self.btn_frame.pack(pady=10)
        
        self.start_btn = ttk.Button(
            self.btn_frame, 
            text="Iniciar", 
            command=self.start_recording
        )
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(
            self.btn_frame, 
            text="Detener", 
            command=self.stop_recording, 
            state=tk.DISABLED
        )
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        # Consola de salida
        self.console = tk.Text(self.master, height=10, width=50)
        self.console.pack(pady=10)

    def start_recording(self):
        self.stop_event.clear()
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        
        selected_model = self.models[self.model_var.get()]
        self.recording_thread = threading.Thread(
            target=self.audio_processing,
            args=(selected_model,),
            daemon=True
        )
        self.recording_thread.start()

    def stop_recording(self):
        self.stop_event.set()
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.log("Proceso detenido")

    def audio_processing(self, model):
        p = pyaudio.PyAudio()
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
        
        self.log("Sistema activo. Detectando voz...")
        
        try:
            while not self.stop_event.is_set():
                data = stream.read(CHUNK, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.int16)
                
                if self.vad(audio_data):
                    frames = [data]
                    self.log("Grabando...")
                    
                    while not self.stop_event.is_set():
                        data = stream.read(CHUNK, exception_on_overflow=False)
                        audio_data = np.frombuffer(data, dtype=np.int16)
                        
                        if self.vad(audio_data):
                            frames.append(data)
                        else:
                            break
                    
                    self.process_audio(frames, model, p)
                    
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

    def vad(self, audio_data):
        # Use the absolute value or a small epsilon to prevent sqrt of negative numbers
        mean_sq = np.mean(audio_data**2)
        print(np.sqrt(abs(mean_sq)))
        return np.sqrt(abs(mean_sq)) > THRESHOLD  # Or np.sqrt(mean_sq + 1e-9) > THRESHOLD

    def process_audio(self, frames, model, p_audio):
        self.log("Procesando audio...")
        wf = wave.open("output.wav", 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p_audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        try:
            y, sr = librosa.load("output.wav", sr=None)
            features = extraer_caracteristicas(y, sr)
            df_features = pd.DataFrame(features, index=[0])
            scaler = MinMaxScaler()
            X_normalized = scaler.fit_transform(df_features)
            prediction = model.predict(X_normalized)[0]
            self.log(f"Predicción: {prediction}")
        except Exception as e:
            self.log(f"Error: {str(e)}")

    def log(self, message):
        self.log_queue.put(message)

    def update_log(self):
        while not self.log_queue.empty():
            msg = self.log_queue.get()
            self.console.insert(tk.END, msg + "\n")
            self.console.see(tk.END)
        self.master.after(100, self.update_log)


if __name__ == "__main__":
    root = tk.Tk()
    app = VoiceRecognitionApp(root)
    root.mainloop()