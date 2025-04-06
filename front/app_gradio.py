# app_gradio.py
import gradio as gr
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Carregar modelo treinado
# model = load_model("modelo_multiclasse.h5")
# model = load_model("../back/modelo_multiclasse.keras")
# model = load_model("C:\Users\User\VSCode_Projects\gradio_050425a\back\modelo_multiclasse.keras")
# model = load_model("/Users/User/VSCode_Projects/gradio_050425a/back/modelo_multiclasse.keras") - erro no compose.yaml
# model = load_model("C://Users//User//VSCode_Projects//gradio_050425a//back//modelo_multiclasse.keras") - erro no compose.yaml
# model = load_model("/app/model/modelo_multiclasse.keras")
# model = load_model("/Users/User/VSCode_Projects/gradio_050425a/back/modelo_multiclasse.keras")
model = load_model("/back/modelo_multiclasse.keras")


# Nomes das classes
classes = ['avião', 'automóvel', 'pássaro', 'gato', 'veado',
           'cachorro', 'sapo', 'cavalo', 'navio', 'caminhão']

# Função de predição
def classificar_imagem(imagem):
    img = imagem.resize((32, 32))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 32, 32, 3)
    pred = model.predict(img_array)[0]
    resultados = {classes[i]: float(pred[i]) for i in range(10)}
    return resultados

# Interface Gradio
interface = gr.Interface(
    fn=classificar_imagem,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="Classificador CIFAR-10",
    description="Faça upload de uma imagem 32x32 para classificação em 10 categorias."
)

interface.launch(server_name="0.0.0.0", server_port=7860)
