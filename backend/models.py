from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    user_input = Column(Text, nullable=False)
    context = Column(Text, nullable=True) # Combined last 3 messages
    toxicity_score = Column(Float, nullable=False)
    prediction_label = Column(String(50), nullable=False) # "Toxic" or "Non-toxic"
    rephrased_output = Column(Text, nullable=True)
    # Store 6-segment probabilities as a JSON string
    segmentation_scores = Column(Text, nullable=True) 
    timestamp = Column(DateTime, default=datetime.utcnow)