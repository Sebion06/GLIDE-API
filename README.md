# Glide-Api

Glide-Api reprezinta o solutie software web care este capabila sa produca compozitii grafice (imagini) intr-un mod automat, folosindu-se de modelul [Glide](https://gpt3demo.com/apps/openai-glide), creat de Open-AI. Utilizandu-se de modulul de python flask, aceasta prezinta o interfata web cu utilizatorul, dar si un REST-API care poate fi utilizat pentru crearea imaginilor in cadrul altor aplicatii software. 

## Continut

`./app` : aplicatia web

`./notebooks` : Jupyter notebook-ul oferit de echipa Open-AI, explicat prin comentarii

`./usage` : un exemplu de utilizare a aplicatiei pe partea de front-end


## Instalare

Pentru instalarea aplicatiei sau a exemplului de folosire, este nevoie de [python3](https://www.python.org/downloads/), preferabil v. 3.10 si modulul pip instalat.

```bash
#Instalarea aplicatiei:
pip install -r /app/requirements.txt

#Instalarea exemplului:
pip install -r /usage/requirements.txt
```

## Utilizarea aplicatiei

Aplicatia poate fi pornita pe un server local, iar apoi accesata din browser, folosind una din comenzile urmatoare, din directorul `app`:

```bash
flask run
#or
python -m flask run --host=127.0.0.1:5000
```

In cazul in care aceasta este gazduita pe un server de cloud, ne putem conecta la acesta, asemator cu aplicatia locala, stiind adresa si port-ul, sau ne putem folosi de apelurile catre API.

### Apeluri catre REST API:

```bash
#Pentru generarea unei imagini (GET request):
curl https://${ADDRESS}:${PORT}/image?size=64&prompt="mountains in the night"

#Pentru modificarea numarului de pasi de difuzie (POST request)
#model de baza
curl -X POST https://${ADDRESS}:${PORT}/basemodel?diffusion_steps=50
#model de upsampling
curl -X POST https://${ADDRESS}:${PORT}/upmodel?diffusion_steps=27
```
#### Parametrii pentru GET request:

`size` - OBLIGATORIU - marimea imaginii: 64, 128, 256, 512, 1024.. 

`prompt` - OBLIGATORIU - promt-ul text dupa care modelul va genera imaginea 

`batch_size` - numarul de exemple imagini returnate, insa in format-ul unei singuri imagini `default=1 `

`guidance_scale` - parametru care are efectul de a aplifica influenta semnalului de conditionare a imaginii `default=3.0`

`upsample_temp` - parametru pentru a controla claritatea imaginii obtinute `default=0.997`

#### Parametrii pentru POST request:

`diffusion_steps` - OBLIGATORIU - numarul de pasi de difuzie al unui model


### Exemple rezultate API

Rezultat pentru GET request:

```json
{
    "data_type": "image/png;base64",
    "gpu_memory_allocated": "1.72 Gb",
    "gpu_memory_reserved": "4.48 Gb",
    "image": "iVBORw0KG...kJggg==",
    "image_size": 64,
    "processing_time": "2.21875 s"
}
```

Rezultat pentru POST request, in functie de modelul apelat:
```json
{
    "base_model_total_parameters": 385030726
}
```
```json
{
    "upsample_model_total_parameters": 398361286
}
```
## Docker
Este pusa la dizpozitie un fisier Docker si un container, pentru a putea lansa aplicatia pe diferite servicii de cloud.

[Docker repository](https://hub.docker.com/layers/232168833/sebion06/glide-api/glideapi/images/sha256-c57c1884055a01896094027dcda843a3ce7ae4f3d20b761a8d64bdfe06e175aa?context=repo)

```bash
#Crearea unei imagini noi
cd app
docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .

#Rularea unui nou container
docker run -d -p ${PORT}:${PORT} --name ${IMAGE_NAME}:${IMAGE_TAG}

```