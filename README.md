# RiceScan

RiceScan is a Flask web app for detecting rice false smut stages from field images using a trained YOLO model.

## Classes

The app is configured for 4 false smut stages:

- `initial`
- `yellow`
- `transition`
- `last`

## Project Structure

```text
ricescan/
  app.py
  index.html
  requirements.txt
  models/
    best.pt
  static/
    uploads/
```

## Model

Place your trained YOLO model here:

```text
models/best.pt
```

The model class names should match the app classes:

```text
initial, yellow, transition, last
```

## Setup

Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000/
```

The Flask app now binds to all network interfaces by default, so other devices
on the same Wi-Fi/LAN can open:

```text
http://YOUR_COMPUTER_IP:5000/
```

On Windows, find your IP with:

```powershell
ipconfig
```

## Nginx reverse proxy

Copy or include `nginx-ricescan.conf` in your Nginx sites/config folder, then
reload Nginx. It proxies browser traffic to the Flask app on port `5000`.

## Live camera over HTTPS

Live camera streaming requires a secure browser context. On another phone,
tablet, or computer, plain `http://YOUR_COMPUTER_IP:5000` will not work for
`Start Live`.

Install the updated dependencies:

```bash
pip install -r requirements.txt
```

Start RiceScan with HTTPS enabled:

```powershell
$env:ENABLE_HTTPS="1"
python app.py
```

Open the printed `https://YOUR_COMPUTER_IP:5443/` URL on the other device.
Because the app creates a local development certificate, the browser may show a
certificate warning the first time. Continue/accept it, then allow camera
permission.

For production, put Nginx in front of the app with a trusted SSL certificate.

## Features

- Single YOLO model support via `models/best.pt`
- Image upload and prediction preview
- Live camera detection from the browser
- Bounding boxes drawn on the result image
- Stage summary, confidence, risk, and action text
- Light/dark mode
- English/Bangla language toggle
