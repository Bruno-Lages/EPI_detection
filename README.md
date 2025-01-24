# Detecção de Veículos e Pessoas Sem Capacete com YOLO
Este script processa um arquivo de vídeo utilizando o modelo de detecção de objetos YOLOv8 para detectar veículos e pessoas. 
Ele identifica veículos e pessoas sem capacetes, salvando as regiões de interesse (ROIs) como arquivos de imagem. 
Além disso, ele registra alertas no log sempre que tais objetos são detectados.

## Observações do algoritmo e modelo treinado

O script utiliza um modelo YOLO treinado com base em imagens capturadas por uma câmera, com anotações manuais focadas em objetos visíveis perto da câmera, ignorando objetos no horizonte. 
Durante o processamento de cada quadro, o modelo detecta diferentes classes de objetos, como veículos, pessoas e capacetes, exibindo as respectivas bounding boxes. 
Cada objeto detectado recebe um ID único para acompanhamento.

Um sistema de rastreamento é implementado para seguir os objetos ao longo dos quadros, utilizando a distância euclidiana entre os centros das bounding boxes para identificar se um objeto detectado no quadro atual corresponde a um objeto rastreado. 
No caso das pessoas e capacetes, há uma verificação de proximidade das caixas de ambos os objetos para identificar pessoas sem capacete, considerando que o capacete deve estar fisicamente sobre a pessoa para que a detecção seja válida.

Sempre que um veículo ou uma nova pessoa é detectado, a região de interesse (ROI) é salva, e um alerta é registrado no log. 
No entanto, devido ao processamento frame a frame e à acurácia reduzida do modelo para detectar capacetes, pode ocorrer que uma pessoa seja alertada como estando sem capacete devido à falha de detecção do capacete em um quadro específico ou a oclusão temporária do mesmo. 
Além disso, a métrica baseada em distância pode causar o reconhecimento de dois objetos da mesma classe como um único objeto se as bounding boxes estiverem muito próximas.

O sistema também lida com oclusões rápidas, permitindo que um objeto seja rastreado mesmo quando oculto por um curto período. 
No entanto, existe um limite de tempo configurável para o máximo de quadros que um objeto pode ser ocluído antes de ser removido do rastreamento, garantindo que o sistema não continue rastreando objetos que não são visíveis por tempo demais.

## Como instalar
Após clonar o repositório, basta baixar os pacotes no `requirements.txt`.

```
pip install -r requirements.txt
```

## Como rodar o código
Basta chamar o arquivo `inference.py` com os parâmetros especificado abaixo

`model_path`: Caminho para o modelo treinado YOLO (arquivo .pt) que será usado para detecção de objetos.
Valor Padrão: "runs/detect/train3/weights/best.pt"

`video_path`: Caminho para o arquivo de vídeo que será processado. O vídeo será analisado quadro a quadro.
Valor Padrão: "data/teste.mp4"

`log_file_path`: Caminho para o arquivo de log onde os alertas serão registrados. Sempre que um veículo ou pessoa sem capacete for detectado, um alerta será adicionado a esse arquivo.
Valor Padrão: "alertas.log"

`vehicle_roi_folder`: Caminho para a pasta onde as regiões de interesse (ROIs) dos veículos detectados serão salvas. As imagens das ROIs serão armazenadas com um nome baseado no timestamp de cada detecção.
Valor Padrão: "ROI/vehicle"

`people_roi_folder`: Caminho para a pasta onde as regiões de interesse (ROIs) das pessoas sem capacete serão salvas. As imagens das ROIs serão armazenadas com um nome baseado no timestamp de cada detecção.
Valor Padrão: "ROI/people"

`max_occlusion_frames`: Número máximo de quadros consecutivos em que um objeto pode ficar oculto (ocluído) antes de ser removido do rastreamento. Esse parâmetro permite lidar com pequenas oclusões sem perder o rastreamento do objeto.
Valor Padrão: 15

Exemplo de chamada do arquivo:
```
python3 detect_objects.py --model_path "runs/detect/train3/weights/best.pt" --video_path "data/teste.mp4" --log_file_path "alertas.log" --vehicle_roi_folder "ROI/vehicle" --people_roi_folder "ROI/people" --max_occlusion_frames 15
```

