#!/usr/bin/env python3
"""
Credit Card Fraud Detection Pipeline

This script provides a consolidated pipeline for credit card fraud detection:
1. Transforms raw transaction data
2. Constructs a heterogeneous graph 
3. Passes the graph through a GNN model for fraud prediction

Usage:
    python credit_card_fraud_predictor.py --input_file path/to/transaction.csv --model_path path/to/model.pth
"""
import warnings
warnings.filterwarnings("ignore")

import os
import pickle
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("fraud_detection")


# ====================================================
# PART 1: DATA TRANSFORMATION 
# ====================================================

class DataTransformer:
    """
    Transforms raw transaction data for credit card fraud detection.
    
    This class applies necessary preprocessing steps to raw transaction data,
    including encoding categorical features, adding statistical features,
    and scaling numerical features.
    """
    
    def __init__(self, artifacts_dir: str = "./artifacts"):
        """
        Initialize the transformation pipeline.
        
        Args:
            artifacts_dir: Directory containing model artifacts
        """
        self.artifacts_dir = Path(artifacts_dir)
        
        # Define paths to artifacts
        self.customer_mapping_path = self.artifacts_dir / "customer_mapping.pkl"
        self.merchant_mapping_path = self.artifacts_dir / "merchant_mapping.pkl"
        self.label_encoders_path = self.artifacts_dir / "label_encoders.pkl"
        self.scaler_path = self.artifacts_dir / "scaler.pkl"
        self.customer_stats_path = self.artifacts_dir / "customer_stats.pkl"
        self.merchant_stats_path = self.artifacts_dir / "merchant_stats.pkl"
        
        # Initialize attributes
        self.customer_mapping: Dict[Any, int] = {}
        self.merchant_mapping: Dict[Any, int] = {}
        self.label_encoders: Dict[str, Any] = {}
        self.scaler: Any = None
        self.customer_stats: Dict[int, Dict[str, float]] = {}
        self.merchant_stats: Dict[int, Dict[str, float]] = {}
        
        self.default_customer_stats = {'min_amt': 0, 'max_amt': 0, 'std_amt': 0}
        self.default_merchant_stats = {'min_amt': 0, 'max_amt': 0, 'std_amt': 0}
        
        # Load artifacts
        self._load_artifacts()
        
        # Define columns to drop and features to scale
        self.columns_to_drop = [
            'Unnamed: 0', 'first', 'last', 'street', 'city', 'zip', 'state',
            'lat', 'long', 'merch_lat', 'merch_long', 'merch_zipcode',
            'job', 'unix_time'
        ]
        
        self.features_to_scale = [
            'amt', 'customer_min_amt', 'customer_max_amt', 'customer_amt_std',
            'merchant_min_amt', 'merchant_max_amt', 'merchant_amt_std', 'amt_per_city_pop'
        ]
        
        self.final_features = [
            'amt', 'category', 'customer_min_amt', 'customer_max_amt', 'customer_amt_std',
            'merchant_min_amt', 'merchant_max_amt', 'merchant_amt_std', 'amt_per_city_pop',
            'trans_hour', 'trans_month', 'day_of_week', 'age', 'gender',
            'customer_id', 'merchant_id', 'transaction_unique'
        ]

    def _load_artifacts(self) -> None:
        """Load all transformation artifacts from disk."""
        try:
            for path, attr, desc in [
                (self.customer_mapping_path, 'customer_mapping', 'Customer mapping'),
                (self.merchant_mapping_path, 'merchant_mapping', 'Merchant mapping'),
                (self.label_encoders_path, 'label_encoders', 'Label encoders'),
                (self.scaler_path, 'scaler', 'Scaler'),
                (self.customer_stats_path, 'customer_stats', 'Customer statistics'),
                (self.merchant_stats_path, 'merchant_stats', 'Merchant statistics')
            ]:
                with open(path, 'rb') as f:
                    setattr(self, attr, pickle.load(f))
                if attr not in ['scaler', 'label_encoders']:
                    logger.info(f"{desc} loaded with {len(getattr(self, attr))} entries")
                else:
                    logger.info(f"{desc} loaded")
            logger.info("All transformation artifacts loaded successfully")
        except Exception as e:
            logger.error(f"Error loading artifacts: {str(e)}")
            raise

    def transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform raw transaction data for fraud detection.
        
        Args:
            df: Raw transaction data as a DataFrame
            
        Returns:
            DataFrame: Transformed data ready for graph construction
        """
        logger.info("Starting data transformation from DataFrame input")
        
        if df.empty:
            logger.error("Data is empty, cannot proceed with transformation")
            return df

        df = self._preprocess_data(df)
        logger.info(f"Data transformation complete. Shape: {df.shape}")
        return df
    # def transform_data(self, data_path: str) -> pd.DataFrame:
    #     """
    #     Transform raw transaction data for fraud detection.
        
    #     Args:
    #         data_path: Path to the raw data file
            
    #     Returns:
    #         DataFrame: Transformed data ready for graph construction
    #     """
    #     logger.info(f"Starting data transformation from {data_path}")
    #     df = self._load_data(data_path)
    #     if df.empty:
    #         logger.error("Data is empty, cannot proceed with transformation")
    #         return df
    #     df = self._preprocess_data(df)
    #     logger.info(f"Data transformation complete. Shape: {df.shape}")
    #     return df

    # def _load_data(self, data) -> pd.DataFrame:
    #     """Load the data from the specified path."""
    #     try:
    #         df = pd.read_csv(data_path)
    #         logger.info(f"Data loaded from {data_path} with shape {df.shape}")
    #         return df
    #     except FileNotFoundError:
    #         logger.error(f"File not found: {data_path}")
    #         return pd.DataFrame()

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all preprocessing steps to the data."""
        df = self._handle_missing_values(df)
        df = df.drop(columns=[col for col in self.columns_to_drop if col in df.columns])
        df = self._process_date_features(df)
        df = self._map_entity_ids(df)
        df = self._encode_categorical_features(df)
        df = self._add_statistics_features(df)
        df = self._scale_features(df)
        df = self._finalize_dataset(df)
        
        logger.info(f"Data preprocessing complete. Final shape: {df.shape}")
        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in data."""
        df['category'] = df['category'].fillna('unknown')
        df['amt'] = pd.to_numeric(df['amt'], errors='coerce').fillna(0)
        if 'city_pop' in df.columns:
            df['city_pop'] = pd.to_numeric(df['city_pop'], errors='coerce').fillna(1)
        logger.info("Missing values handled in data")
        return df

    def _process_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process date-related features."""
        date_cols = ['trans_date_trans_time', 'dob']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        if 'trans_date_trans_time' in df.columns:
            df['trans_hour'] = df['trans_date_trans_time'].dt.hour
            df['trans_month'] = df['trans_date_trans_time'].dt.month
            df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek
            df['trans_year'] = df['trans_date_trans_time'].dt.year
        else:
            df['trans_hour'] = 12
            df['trans_month'] = 1
            df['day_of_week'] = 0
            df['trans_year'] = 2020
        
        if 'dob' in df.columns and 'trans_date_trans_time' in df.columns:
            df['birth_year'] = df['dob'].dt.year.fillna(1990)
            df['birth_month'] = df['dob'].dt.month.fillna(1)
            df['birth_day'] = df['dob'].dt.day.fillna(1)
            df['trans_day'] = df['trans_date_trans_time'].dt.day
            df['age'] = df['trans_year'] - df['birth_year']
            not_had_birthday = ((df['trans_month'] < df['birth_month']) | 
                                ((df['trans_month'] == df['birth_month']) & 
                                 (df['trans_day'] < df['birth_day'])))
            df.loc[not_had_birthday, 'age'] -= 1
            df['age'] = df['age'].apply(lambda x: max(0, min(x, 100)) if pd.notna(x) else 30)
            df = df.drop(columns=['birth_year', 'birth_month', 'birth_day', 'trans_day', 'trans_year'])
        else:
            df['age'] = 30
        
        if 'trans_date_trans_time' in df.columns:
            df = df.drop(columns=['trans_date_trans_time'])
        
        logger.info("Date features processed")
        return df

    def _map_entity_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map customer and merchant to their IDs from training, assigning new IDs for unseen entities."""
        if 'cc_num' in df.columns:
            max_customer_id = max(self.customer_mapping.values(), default=999999)
            customer_id_mapping = {cc_num: self.customer_mapping.get(cc_num, max_customer_id + 1 + i)
                                   for i, cc_num in enumerate(df['cc_num'].unique())}
            df['customer_id'] = df['cc_num'].map(customer_id_mapping)
            df = df.drop(columns=['cc_num'])
            logger.info(f"Mapped {len(customer_id_mapping)} customers")
        
        if 'merchant' in df.columns:
            max_merchant_id = max(self.merchant_mapping.values(), default=999999)
            merchant_id_mapping = {merchant: self.merchant_mapping.get(merchant, max_merchant_id + 1 + i)
                                   for i, merchant in enumerate(df['merchant'].unique())}
            df['merchant_id'] = df['merchant'].map(merchant_id_mapping)
            df = df.drop(columns=['merchant'])
            logger.info(f"Mapped {len(merchant_id_mapping)} merchants")
        
        return df

    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features using pre-trained encoders."""
        if 'gender' in df.columns and 'gender_mapping' in self.label_encoders:
            gender_mapping = self.label_encoders['gender_mapping']
            df['gender'] = df['gender'].map(gender_mapping).fillna(-1).astype(int)
        
        if 'category' in df.columns and 'category_mapping' in self.label_encoders:
            category_mapping = self.label_encoders['category_mapping']
            df['category'] = df['category'].astype(str).map(lambda x: category_mapping.get(x, 0)).astype(int)
        
        logger.info("Categorical features encoded")
        return df

    def _add_statistics_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add customer and merchant statistics features using pre-computed stats."""
        if 'customer_id' in df.columns:
            df['customer_min_amt'] = df['customer_id'].apply(
                lambda cid: self.customer_stats.get(cid, self.default_customer_stats)['min_amt']
            )
            df['customer_max_amt'] = df['customer_id'].apply(
                lambda cid: self.customer_stats.get(cid, self.default_customer_stats)['max_amt']
            )
            df['customer_amt_std'] = df['customer_id'].apply(
                lambda cid: self.customer_stats.get(cid, self.default_customer_stats)['std_amt']
            )
        
        if 'merchant_id' in df.columns:
            df['merchant_min_amt'] = df['merchant_id'].apply(
                lambda mid: self.merchant_stats.get(mid, self.default_merchant_stats)['min_amt']
            )
            df['merchant_max_amt'] = df['merchant_id'].apply(
                lambda mid: self.merchant_stats.get(mid, self.default_merchant_stats)['max_amt']
            )
            df['merchant_amt_std'] = df['merchant_id'].apply(
                lambda mid: self.merchant_stats.get(mid, self.default_merchant_stats)['std_amt']
            )
        
        if 'amt' in df.columns and 'city_pop' in df.columns:
            df['amt_per_city_pop'] = (df["amt"] / (df["city_pop"] + 1)).round(6)
        else:
            df['amt_per_city_pop'] = 0.0
        
        df['transaction_unique'] = range(2000000, 2000000 + len(df))
        
        logger.info("Statistics features added")
        return df

    def _scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale numerical features using the pre-trained scaler."""
        available_scale_features = [col for col in self.features_to_scale if col in df.columns]
        if available_scale_features and self.scaler:
            for col in available_scale_features:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            try:
                scaled_values = self.scaler.transform(df[available_scale_features])
                df[available_scale_features] = np.round(scaled_values, 6)
            except Exception as e:
                logger.error(f"Error scaling features: {str(e)}")
                raise
        logger.info(f"Scaled {len(available_scale_features)} features")
        return df

    def _finalize_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select final features and ensure consistent data types."""
        final_features = self.final_features.copy()
        if 'is_fraud' in df.columns:
            final_features.append('is_fraud')
        available_features = [col for col in final_features if col in df.columns]
        df = df[available_features]
        
        for col in self.final_features:
            if col not in df.columns:
                df[col] = -1 if col in ['category', 'gender', 'day_of_week'] else 0
                logger.warning(f"Added missing feature '{col}' with default values")
        
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        float_cols = df.select_dtypes(include=['float']).columns
        if len(float_cols) > 0:
            df[float_cols] = df[float_cols].round(6)
        
        logger.info(f"Finalized dataset with {len(df.columns)} features")
        return df


# ====================================================
# PART 2: GRAPH CONSTRUCTION
# ====================================================

class GraphConstructor:
    """
    Constructs a heterogeneous graph from the transformed transaction data.
    The graph consists of 'customer', 'merchant', and 'transaction' nodes,
    with edges representing the relationships between them.
    """
    
    def create_node_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates unique node IDs for customers, merchants, and transactions.
        
        Args:
            df: Input DataFrame containing transaction data
            
        Returns:
            DataFrame with added node ID columns
        """
        df["transaction_node"] = df["transaction_unique"].astype(int)
        df["customer_node"] = df["customer_id"].astype(int)
        df["merchant_node"] = df["merchant_id"].astype(int)
        df.drop(columns=["customer_id", "merchant_id", "transaction_unique"], inplace=True)
        return df

    def create_node_features(self, df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Creates feature tensors for customer, merchant, and transaction nodes.
        
        Args:
            df: DataFrame with transaction data and node IDs
            
        Returns:
            Tuple of tensors for customer, merchant, and transaction features
        """
        # Customer features
        customer_features_list = ["customer_min_amt", "customer_max_amt", "customer_amt_std", "age", "gender"]

        # Merchant features
        merchant_features_list = ["merchant_min_amt", "merchant_max_amt", "merchant_amt_std"]

        # Transaction features
        transaction_features_list = ["amt", "amt_per_city_pop", "trans_hour", "trans_month", "day_of_week", "category"]

        customer_features_dim = len(customer_features_list)
        merchant_features_dim = len(merchant_features_list)
        transaction_features_dim = len(transaction_features_list)

        unique_customer_nodes = df["customer_node"].unique()
        unique_merchant_nodes = df["merchant_node"].unique()

        customer_features = torch.zeros((len(unique_customer_nodes), customer_features_dim), dtype=torch.float32)
        merchant_features = torch.zeros((len(unique_merchant_nodes), merchant_features_dim), dtype=torch.float32)
        transaction_features = torch.tensor(df[transaction_features_list].values, dtype=torch.float32)

        for i, customer_id in enumerate(unique_customer_nodes):
            group = df[df["customer_node"] == customer_id]
            customer_features[i] = torch.tensor(group[customer_features_list].mean().values, dtype=torch.float32)

        for i, merchant_id in enumerate(unique_merchant_nodes):
            group = df[df["merchant_node"] == merchant_id]
            merchant_features[i] = torch.tensor(group[merchant_features_list].mean().values, dtype=torch.float32)

        return customer_features, merchant_features, transaction_features

    def construct_graph(self, df: pd.DataFrame) -> tuple[HeteroData, list]:
        """
        Constructs the heterogeneous graph from the transformed data.
        
        Args:
            df: DataFrame containing the transformed data
            
        Returns:
            Tuple containing the constructed graph and transaction indices
        """
        logger.info("Starting graph construction")
        
        # Assign node IDs
        df = self.create_node_ids(df)

        # Get unique nodes for each type
        unique_customers = df["customer_node"].unique()
        unique_merchants = df["merchant_node"].unique()

        # Create mappings from original IDs to 0-based indices
        customer_id_to_index = {id_: idx for idx, id_ in enumerate(unique_customers)}
        merchant_id_to_index = {id_: idx for idx, id_ in enumerate(unique_merchants)}
        transaction_id_to_index = {df["transaction_node"].iloc[i]: i for i in range(len(df))}

        # Create edge indices using mapped IDs
        customer_to_transaction_edges = torch.tensor([
            [customer_id_to_index[cust] for cust in df["customer_node"]],
            [transaction_id_to_index[trans] for trans in df["transaction_node"]]
        ], dtype=torch.long)

        transaction_to_merchant_edges = torch.tensor([
            [transaction_id_to_index[trans] for trans in df["transaction_node"]],
            [merchant_id_to_index[merch] for merch in df["merchant_node"]]
        ], dtype=torch.long)

        # Create node features
        customer_features, merchant_features, transaction_features = self.create_node_features(df)

        # Prepare transaction indices for reference
        transaction_indices = [transaction_id_to_index[trans] for trans in df["transaction_node"]]

        # Assemble the HeteroData object
        data = HeteroData()
        data["customer"].x = customer_features
        data["merchant"].x = merchant_features
        data["transaction"].x = transaction_features
        data["customer", "transacts", "transaction"].edge_index = customer_to_transaction_edges
        data["transaction", "occurs_at", "merchant"].edge_index = transaction_to_merchant_edges
        # Add reverse edges
        data["transaction", "transacted_by", "customer"].edge_index = customer_to_transaction_edges.flip(0)
        data["merchant", "related_to", "transaction"].edge_index = transaction_to_merchant_edges.flip(0)
        
        # Store original node IDs (optional, for reference)
        data["customer"].n_id = torch.tensor(unique_customers)
        data["merchant"].n_id = torch.tensor(unique_merchants)
        data["transaction"].n_id = torch.tensor(df["transaction_node"].values)

        logger.info("Graph construction complete")
        return data, transaction_indices


# ====================================================
# PART 3: GNN MODEL AND PREDICTION
# ====================================================

class GNN(torch.nn.Module):
    """
    Graph Neural Network model for heterogeneous graphs.
    It consists of multiple HeteroConv layers with SAGEConv for each edge type
    and a final linear layer for prediction.
    """
    def __init__(self, metadata, hidden_dim):
        """
        Initialize the GNN model.
        
        Args:
            metadata: Metadata of the heterogeneous graph (node types, edge types)
            hidden_dim: Dimension of the hidden layers
        """
        super(GNN, self).__init__()
        self.conv1 = HeteroConv({edge_type: SAGEConv((-1, -1), hidden_dim) for edge_type in metadata[1]}, aggr='mean')
        self.conv2 = HeteroConv({edge_type: SAGEConv((-1, -1), hidden_dim) for edge_type in metadata[1]}, aggr='mean')
        self.conv3 = HeteroConv({edge_type: SAGEConv((-1, -1), hidden_dim) for edge_type in metadata[1]}, aggr='mean')
        self.conv4 = HeteroConv({edge_type: SAGEConv((-1, -1), hidden_dim) for edge_type in metadata[1]}, aggr='mean') 
        self.conv5 = HeteroConv({edge_type: SAGEConv((-1, -1), hidden_dim) for edge_type in metadata[1]}, aggr='mean') 
        self.lin = torch.nn.Linear(hidden_dim, 1)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x_dict, edge_index_dict):
        """
        Forward pass of the GNN model.
        
        Args:
            x_dict: Dictionary of node features for each node type
            edge_index_dict: Dictionary of edge indices for each edge type
            
        Returns:
            Output logits for the 'transaction' nodes
        """
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = {key: self.dropout(x) for key, x in x_dict.items()}

        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = {key: self.dropout(x) for key, x in x_dict.items()}

        x_dict = self.conv3(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = {key: self.dropout(x) for key, x in x_dict.items()}

        x_dict = self.conv4(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = {key: self.dropout(x) for key, x in x_dict.items()}

        x_dict = self.conv5(x_dict, edge_index_dict)

        return self.lin(x_dict["transaction"]).squeeze(-1)


class FraudPredictor:
    """
    Class to use the trained GNN model to predict fraud probability.
    It loads the model, performs inference, and returns the fraud probability.
    """
    def __init__(self, model_path: str, hidden_channels: int = 64):
        """
        Initialize the FraudPredictor.
        
        Args:
            model_path: Path to the saved model
            hidden_channels: Number of hidden channels in the GNN
        """
        self.model_path = model_path
        self.hidden_channels = hidden_channels
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

    def predict(self, data: HeteroData) -> dict:
        """
        Predict fraud probability for transactions.
        
        Args:
            data: The heterogeneous graph data
            
        Returns:
            Dictionary with fraud probability and prediction label
        """
        logger.info("Loading model and making prediction...")
        
        # Move data to device
        data = data.to(self.device)
        
        # Load model
        metadata = (list(data.x_dict.keys()), list(data.edge_index_dict.keys()))
        model = GNN(metadata, hidden_dim=self.hidden_channels).to(self.device)
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model.eval()
        
        # Make prediction
        with torch.no_grad():
            logits = model(data.x_dict, data.edge_index_dict)
            probs = torch.sigmoid(logits).cpu().numpy()
            
        # For a single transaction
        if len(probs) == 1:
            prob = float(probs[0])
            prediction = "Fraudulent" if prob > 0.5 else "Legitimate"
            logger.info(f"Prediction: {prediction} (Probability: {prob:.4f})")
            return {"fraud_probability": prob, "prediction": prediction}
        # For multiple transactions
        else:
            results = []
            for i, prob in enumerate(probs):
                prediction = "Fraudulent" if prob > 0.5 else "Legitimate"
                results.append({"transaction_idx": i, "fraud_probability": float(prob), "prediction": prediction})
            logger.info(f"Made predictions for {len(results)} transactions")
            return {"transactions": results}


# ====================================================
# MAIN PIPELINE
# ====================================================

def fraud_detection_pipeline(
    input_data: pd.DataFrame, 
    model_path: str,
    artifacts_dir: str,
    hidden_channels: int = 32
) -> dict:
    """
    Run the full credit card fraud detection pipeline.
    
    Args:
        input_file: Path to raw transaction data
        model_path: Path to trained model weights
        artifacts_dir: Directory containing preprocessing artifacts
        hidden_channels: Number of hidden channels in the GNN
    
    Returns:
        Dictionary with fraud prediction results
    """
    try:
        # Step 1: Transform data
        logger.info("Step 1: Transforming data...")
        transformer = DataTransformer(artifacts_dir=artifacts_dir)
        transformed_data = transformer.transform_data(input_data)
        if transformed_data.empty:
            return {"error": "No data available for prediction"}
            
        # Step 2: Construct graph
        logger.info("Step 2: Constructing graph...")
        graph_constructor = GraphConstructor()
        graph_data, transaction_indices = graph_constructor.construct_graph(transformed_data)
        
        # Step 3: Run prediction
        logger.info("Step 3: Running prediction...")
        predictor = FraudPredictor(model_path=model_path, hidden_channels=hidden_channels)
        prediction_result = predictor.predict(graph_data)
        
        return prediction_result
        
    except Exception as e:
        logger.error(f"Error in fraud detection pipeline: {e}", exc_info=True)
        return {"error": str(e)}


# if __name__ == "__main__":


#     result = fraud_detection_pipeline (
#         input_file="artifacts_dir\predict_data.csv",
#         model_path="artifacts_dir\model.pt",
#         artifacts_dir="artifacts_dir",
#         hidden_channels=32
#     )
    
#     print("\nPrediction Result:")
#     print(f"  {'Fraud Probability:':<20} {result.get('fraud_probability', 'N/A'):.6f}")
#     print(f"  {'Classification:':<20} {result.get('prediction', 'N/A')}")