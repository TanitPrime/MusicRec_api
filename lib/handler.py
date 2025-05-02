import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from thefuzz import fuzz, process


class Fetcher:
    """
    Class responsible for Pinecone and song metadata queries
    """
    def connect(self, api_key, table= "features"):
        """
        Connect to Pinecone 
        args: 
            api_key: secret pinecone api key (from .ENV)
        returns index (works like a table)
        """
        pc = Pinecone(api_key=api_key)
        return pc.Index(table)

    
    
    def fetch_vector_by_id(self,index, track_id: str):
        """
        Fetch a stored vector from Pinecone by its ID.
        args:
            index: Pinecone connection 
            track_id: track ID to query Pinecone with
        Returns the vector as a list of matrices of floats[19, 384].
        """
        response = index.fetch(ids=[track_id])
        if track_id not in response.vectors[track_id].id:
            raise ValueError(f"Track ID '{track_id}' not found in index.")
        print(response)
        return self.reshape_vectors([response.vectors[track_id].values])


    def fetch_vectors_by_ids(self, index, track_ids):
        """
        Fetch stored vectors from Pinecone by their IDs.
        args:
            index: Pinecone table
            track_ids: list of track IDs
        Returns list of vectors for each ID.
        """
        try:
            # Fetch
            response = index.fetch(ids=track_ids)
            # Double check for inconsistencies
            for track_id in track_ids:
                if track_id not in response.vectors[track_id].id:
                    raise ValueError(f"Track ID '{track_id}' not found in index.")
            print(f"✓ Found {len(response.vectors.values())} matches")
            return self.reshape_vectors([value.values for value in  response.vectors.values()])
        except Exception as e:
            print(f"× Query failed: {str(e)}")
        raise



    def reshape_vectors(self, vectors, segment_size=384):
        """
        Reshape vector into a 2d matrix matching song features shape of [19,384]
        args:
            vectors: list of vectors
        """
        return [[vector[i:i+segment_size] for i in range(0, len(vector), segment_size)] for vector in vectors]
    
    def flatten_vectors(self, features):
        """
        Flatten vector 1D to match pinecone search
        args:
            features: list or tensor of song featyres [19,384]
        """
        # Check if input is a tensor or not beforehand
        if torch.is_tensor(features):
            return features.flatten().tolist()
        return torch.tensor(features, dtype= torch.float32).flatten().tolist()


    def fetch_vector_by_cosine(self, index, features, top_k= 5):
        """
        Query Pinecone using a full feature vector (flattened), 
        and return the top_k most similar items.
        
        input_vector must be a flat list of floats.
        """
        query_result = index.query(
            vector= features,
            top_k=top_k,
            include_metadata=False  # Change to True if you want metadata
        )
        return query_result["matches"]
    
    def _combine_row_with_weights(self, row, columns, weights):
        """Helper to combine row data with optional weights."""
        if weights:
            return ' '.join(
                str(row[col]) 
                for col, weight in zip(columns, weights) 
                for _ in range(int(weight))  # Repeat based on weight
            )

    def init_combined_song_string(self,df, columns, weights):
        # Generate combined strings for fast search
        combined_matches=  df.apply(
            lambda row: self._combine_row_with_weights(row, columns, weights),
            axis=1
        ).tolist()
        return combined_matches
    
    def get_fuzzy_matches(self, df, combined_strings, search_term, k=5):
        """
        Returns track_ids of top K matches without modifying the DataFrame.
        
        Args:
            df: Song metadata DataFrame (read-only).
            search_term: Query string.
            columns: Columns to search (e.g., ['artists', 'track_name', 'album_name']).
            weights: Optional list of weights for each column (e.g., [1, 2, 0.5]).
            k: Number of results. ooga booga
        """

        # Get top matches
        matches = process.extractBests(
            search_term, 
            combined_strings, 
            limit=k, 
            scorer=fuzz.token_set_ratio
        )
        print(f"found matches {matches}")
        # Recover original indices by matching strings
        matched_indices = [
            combined_strings.index(match[0])  # Find index of matched string
            for match in matches
        ]
        
        #return df.iloc[matched_indices]["track_id"].tolist() 
        return df.iloc[matched_indices].to_dict("records")


####################################################################################################


class Preprocessor:
    """
    Class responsible for preprocessing playlist
    """
    def __init__(self, average_name_embs= None, sequence_length = 5):
        self.sequence_length = sequence_length
        self.average_name_embs = average_name_embs
        self.transformer= SentenceTransformer("all-MiniLM-L6-v2")

    def transform_playlist(self, features):
        """
        Transform raw song features into appropriate format
        args:
            features: dict {ID: [ metadata + song features]}
        return list of features [19 ,384]
        """
        # Todo: sentence transform metadata
        # upscale scalars
        # normalize
        # iterate on every song

        return None
    

    def encode_and_normalise(self, playlist_name):
        """ 
        Sentence Transform playlist name and normalise
        args: 
            playlist_name: Name or title of the playlist
        return name embedding [384]
        """
        embeddings= self.transformer.encode(playlist_name)
        norm = np.linalg.norm(embeddings)
        return embeddings / norm if norm != 0 else embeddings

    def preprocess_playlist(self, features, playlist_name):
        """ 
        Preprocess playlist into sequences and targets and name embeddings to be ready for the model
            args:
                features: list of features
                name: playlist name. Use average as default
            Returns:
                name_embs: tensor of playlist name embedding cast to length of sequences
                sequences: tensor of input sequences [batch, sequence_length=5, features= 19, cell= 384]
                targets: tensor of targets [batch, 19, 384]
        """
        # Make sure features is at least 5 songs long
        if len(features) < 5:
            raise Exception(f"Features are less than 5")

        # Make sure features is a tensor
        if not torch.is_tensor(features):
            features = torch.tensor(features, dtype= torch.float32)

        # Name embedding
        if not self.average_name_embs:
            playlist_name_embs= self.encode_and_normalise(playlist_name)

        # Make sequences and targets and embs
        sequences= []
        targets= []
        for i in range(len(features) - self.sequence_length):
            input_sequence = [song_features for song_features in features[i:i + self.sequence_length]]
            sequence= np.array(input_sequence)
            sequences.append(sequence)

            target= features[i + self.sequence_length]
            targets.append(np.array(target))

        print(f"{len(sequences)} sequences made from '{playlist_name}' playlist")
        return (torch.tensor(np.array([playlist_name_embs for _ in range(len(sequences))]), dtype=torch.float32),
                torch.tensor(sequences, dtype=torch.float32),
                torch.tensor(targets, dtype=torch.float32))