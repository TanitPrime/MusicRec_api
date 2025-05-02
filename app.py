import json
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from flask import Flask, request, jsonify
import sys
from flask_cors import CORS
from pathlib import Path 

sys.path.append(str(Path.cwd()))

# Load api key
key = os.getenv("KEY")  # Works if set in hosting platform's dashboard
if not key:
    load_dotenv()      # Fallback to .env file (local only)
    key = os.getenv("KEY")

# Import metadata
metadata = pd.read_feather("features_metadata.feather")
metadata= metadata.reset_index(drop= True)

from lib.handler import Preprocessor, Fetcher
# Connect to Pinecone
fetcher= Fetcher()
index = fetcher.connect(key)
# Generate combined string on start up for faster lookup
commbined_string = fetcher.init_combined_song_string(metadata,["track_name", "artists", "album_name"] ,[2 , 1 , 1])
# Instantiate preprocessor object
prep= Preprocessor()

from lib.recommend import Recommend
rc= Recommend()
# Load model weights
model= rc.init_model("best_model.pth")

app = Flask(__name__)


# Load preloaded playlists 
with open("clean_playlists.json", "r") as f:
    playlists= json.load(f)


# Use this when generating recommendations
def get_recommendations(predictions, pinecone_index, input_song_ids, top_k_per_seq=3):
    """
    Function to find songs excluding the ones already in the playlist
    args:
        predictions: list predicted song IDs
        pinecone_index: Pinecone table
        input_song_ids: list of input playlist IDs
        top_k_per_seq: int of how many top similar songs per sequence to fetch from Pinecone
    Returns IDs of recommended songs, excluding songs already in the playlist.
    """
    all_recs = []
    for pred in predictions:  # Each pred shape: [19, 384]
        # Query Pinecone
        recs = pinecone_index.query(
            vector=pred.flatten().tolist(),
            top_k=top_k_per_seq * 2,  # Query extra to account for filtering
            include_values=False
        )
        # Filter out songs already in the playlist
        filtered_recs = [
            match.id for match in recs.matches 
            if match.id not in input_song_ids
        ]
        all_recs.extend(filtered_recs[:top_k_per_seq])  # Keep only top-k after filtering

    return all_recs

# Map IDs to meta
def ids_to_metadata(metadata, ids):
    """
    Function to get metadata of songs from IDs
    args: 
        metadata: dataframe of songs metadata (artists, album name, track name)
        ids: list of song IDs
    returns: a filtered dataframe with matching metadata
    """
    return metadata[metadata["track_id"].isin(ids)].drop_duplicates()


# App routes begin here
@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

# Pagination route
@app.route("/get_songs_by_ids", methods= ["POST"]) # Takes list of IDs works with list of a singular ID
def get_songs_by_ids():
    data = request.json
    if not data.get("ids"):
        return jsonify({"error": "Playlist IDs are required"}), 400
    try:
        return jsonify({"song": ids_to_metadata(metadata, data["id"]).to_dict("records")}) 
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Fuzzy search route
@app.route("/search", methods= ["POST"]) # Accepts search term and spotify connection flag
def get_song_by_meta():
    data = request.json
    if not data.get("term"):
        return jsonify({"error": "Search term is required"}), 400
    try:
        if not data["spotify_connection"]:
            # do fuzzy match on all features
            matches = fetcher.get_fuzzy_matches(metadata, commbined_string, data["term"])
            return jsonify({"matches": matches}) # method already returns a dict in this case
        else:
            # search using spotify api
            pass
    except Exception as e:
        return jsonify({"error": str(e)}), 500   

# Generate recommendations point
@app.route("/recommend", methods= ["POST"])
def rec():
    data = request.json
    if not data.get("ids"):
        return jsonify({"error": "Playlist IDs are required"}), 400
    
    app.logger.info(f"Recieved playlst {data['name']} with {len(data['ids'])} songs")

    try:
        # Fetch features from Pinecone
        features = fetcher.fetch_vectors_by_ids(index, data["ids"])
        if not features:
            return jsonify({"error": "No songs found in database"}), 404
        
        #Log success
        app.logger.info("features retrieved successully")

        # Preprocess and construct sequences
        name_embs, seq , targets= prep.preprocess_playlist(features, data["name"])
        # Predict features for each sequence
        results= rc.predict_playlist(model, seq, name_embs, targets)
        if results:
            app.logger.info(f"Generated  {len(results['predictions'])} predictions")
        # Fetch top k similar features for each prediction
        id_matches = get_recommendations(results["predictions"],index,data["ids"])
        # Map predictions to metadata
        recommendations = ids_to_metadata(metadata, id_matches)
        # Return response
        
        return jsonify({"recommendations": recommendations.to_dict("records")})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Pagination point
@app.route('/all-songs', methods= ["POST"])
def get_all_songs():
    data = request.json
    if not data.get("page"):
        return jsonify({"error": "Page number is required"}), 400
    try:
        page = data["page"]
        per_page = 50  # Adjust based on performance testing
        songs = metadata[(page-1)*per_page : page*per_page] 
        return jsonify(songs.to_dict('records'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500  


# Standard error handling
@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

# Run app
CORS(app)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))