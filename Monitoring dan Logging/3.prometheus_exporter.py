from flask import Flask, request, jsonify, Response
import requests
import time
import psutil 
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

app = Flask(__name__)

# Metrik untuk API model
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP Requests')  # Total request yang diterima
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP Request Latency')  # Waktu respons API
FAILED_REQUESTS = Counter('http_requests_failed_total', 'Total number of failed HTTP requests')


# Metrik untuk sistem
CPU_USAGE = Gauge('system_cpu_usage', 'CPU Usage Percentage')  # Penggunaan CPU
RAM_USAGE = Gauge('system_ram_usage', 'RAM Usage Percentage')  # Penggunaan RAM

 
#endpoint for Prometheus metrics
@app.route('/metrics', methods=['GET'])
def metrics():

    #Update system metrics
    CPU_USAGE.set(psutil.cpu_percent(interval=1))
    RAM_USAGE.set(psutil.virtual_memory().percent)
    
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

# Endpoint untuk mengakses API model dan mencatat metrik
@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    REQUEST_COUNT.inc()  # Tambah jumlah request
    

    # Kirim request ke API model
    api_url = "http://127.0.0.1:5002/invocations"
    data = request.get_json()
 
    try:
        response = requests.post(api_url, json=data)
        
        # Periksa status DULU sebelum melanjutkan
        response.raise_for_status()
        
        # Jika berhasil, catat latensi dan kembalikan respons
        duration = time.time() - start_time
        REQUEST_LATENCY.observe(duration) 
        
        return jsonify(response.json())
    
    except Exception as e:
        # Jika terjadi error apapun, cukup inkremen counter FAILED_REQUESTS
        FAILED_REQUESTS.inc() 
        
        # Mengembalikan pesan error ke client
        error_message = {"error": str(e)}
        status_code = 500
        if isinstance(e, requests.exceptions.RequestException):
            status_code = 502 # Bad Gateway
        
        return jsonify(error_message), status_code

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000)