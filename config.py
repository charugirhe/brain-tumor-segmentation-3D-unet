# config.py
# This file reads all settings from the .env file
# No secrets are hardcoded here!

import os
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# Gemini AI API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Database settings
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "brain_tumor_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "root")

# BraTS dataset path
BRATS_DATASET_PATH = os.getenv("BRATS_DATASET_PATH", "C:\\Users\\Dell\\Desktop\\brain_tumor_project\\dataset")

# Build the database connection string
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Model path
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "best_model.pth")

