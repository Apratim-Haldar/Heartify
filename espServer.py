from flask import Flask, request, jsonify
from pymongo import MongoClient
from datetime import datetime

app = Flask(__name__)

# Replace with your MongoDB connection string
client = MongoClient("mongodb+srv://Apratim:wiINvlvnkfc5cRw4@atlascluster.mz70pny.mongodb.net/")  # MongoDB Atlas or local connection
db = client["MaxBPM"]  # Replace with your database name
collection = db["HeartRateMonitor"]  # Replace with your collection name

@app.route('/insertData', methods=['POST'])
def insert_data():
    data = request.get_json()
    max_bpm = data.get('maxbpm')
    avg_hr = data.get('av6')
    min_bpm = data.get('minbpm')
    
    if max_bpm and avg_hr and min_bpm:
        # Insert data into MongoDB as a single document
        document = {
            "maxBPM": int(max_bpm),
            "avgBPM": int(avg_hr),
            "minBPM": int(min_bpm),
            "timestamp": datetime.now()
        }
        collection.insert_one(document)
        return jsonify({"message": "Data inserted successfully!"}), 200
    else:
        return jsonify({"error": "Data not provided"}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # Run the app on all interfaces