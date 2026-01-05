# Import necessary modules and classes
from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, JSON, Boolean, DateTime
import sqlalchemy
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel
from typing import List
import json
from api_handler import call_gpts_concurrently, parse_video_data
from openai import AsyncOpenAI
from urllib.parse import urlparse, parse_qs
from datetime import datetime, timezone
# FastAPI app instance

# TODO: RESPONSE TYPES AND CODES

with open('data/config.json') as jf:
    keys = json.load(jf)

yt_key = keys['YT_API_KEY']
gpt_key = keys['GPT_API_KEY']

with open('data/base_prompt_creator.txt', 'r', encoding='utf-8') as pf:
    creator_prompt = pf.read()
with open('data/base_prompt_comments.txt', 'r', encoding='utf-8') as pf:
    comment_prompt = pf.read()


client = AsyncOpenAI(api_key=gpt_key)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database setup
DATABASE_URL = "sqlite:///./test.db"


engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = sqlalchemy.orm.declarative_base()

class Questions(Base):
    __tablename__ = "SurveyQuestions"

    question_id = Column(Integer, primary_key=True, index=True)
    text = Column(String, nullable=False)
    type = Column(String, nullable=False)
    options = Column(JSON, nullable=False)



class Analytics(Base):
    __tablename__ = "ToolAnalytics"

    interaction_id      = Column(String, primary_key=True)
    user_id             = Column(String, nullable=False)
    video_url           = Column(String, nullable=False)
    video_title         = Column(String, nullable=False)
    video_description   = Column(String, nullable=False)
    video_tags          = Column(JSON, nullable=False)
    video_publisher     = Column(String, nullable=False)
    time_requested      = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable= False)
    platform_label            = Column(String, nullable = True)
    total_time_spent          = Column(Integer, nullable=True)
    time_spent_platform_label = Column(Integer, nullable=True)
    time_spent_creator_label  = Column(Integer, nullable=True)
    time_spent_comment_label  = Column(Integer, nullable=True)
    time_spent_risk_label     = Column(Integer, nullable=True)
    time_spent_recommendation = Column(Integer, nullable=True)
    audio_warning             = Column(Boolean, nullable=True)

# Create tables
Base.metadata.create_all(bind=engine)

# Dependency to get the database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class QuestionsCreate(BaseModel):
    text: str
    type: str
    options: List[str]

class QuestionsResponse(BaseModel):
    id: int
    text: str
    type: str
    options: List[str]

    class Config:
        from_attributes = True  # required for SQLAlchemy → Pydantic

class AnalyticsUpdate(BaseModel):
    platform_label  : str | None
    total_time_spent : int | None
    time_spent_platform_label : int | None
    time_spent_creator_label : int | None
    time_spent_comment_label : int | None
    time_spent_risk_label : int | None
    time_spent_recommendation : int | None
    audio_warning: bool | None


class AnalyticsResponse(BaseModel):
    interaction_id  : str
    user_id         : str
    video_url       : str
    video_title     : str
    video_description: str
    video_tags      : List[str]
    video_publisher : str
    time_requested  : datetime
    platform_label  : str | None
    total_time_spent : int | None
    time_spent_platform_label : int | None
    time_spent_creator_label : int | None
    time_spent_comment_label : int | None
    time_spent_risk_label : int | None
    time_spent_recommendation : int | None
    audio_warning: bool | None

    class Config:
        from_attributes = True  # required for SQLAlchemy → Pydantic



@app.post("/questions/", response_model=QuestionsResponse)
def create_question(
    question: QuestionsCreate,
    db: Session = Depends(get_db)
):
    db_question = Questions(**question.model_dump())
    db.add(db_question)
    db.commit()
    db.refresh(db_question)
    return db_question


@app.get("/questions/", response_model=List[QuestionsResponse])
def get_questions(
    db: Session = Depends(get_db)
):
    return db.query(Questions).all()


@app.get("/questions/{question_id}", response_model=QuestionsResponse)
def get_question(
    question_id: int,
    db: Session = Depends(get_db)
):
    question = db.query(Questions).filter(Questions.question_id == question_id).first()
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")
    return question


@app.put("/questions/{question_id}", response_model=QuestionsResponse)
def update_question(
    question_id: int,
    updated: QuestionsCreate,
    db: Session = Depends(get_db)
):
    question = db.query(Questions).filter(Questions.question_id == question_id).first()
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")

    for key, value in updated.model_dump().items():
        setattr(question, key, value)

    db.commit()
    db.refresh(question)
    return question


@app.delete("/questions/{question_id}")
def delete_question(
    question_id: int,
    db: Session = Depends(get_db)
):
    question = db.query(Questions).filter(Questions.question_id == question_id).first()
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")

    db.delete(question)
    db.commit()
    return {"message": "Question deleted successfully"}

def create_base_interaction(data: dict, user_id: str, url: str, interaction_id: str, db: Session):
    db_interaction = Analytics(
        user_id = user_id,
        interaction_id = interaction_id,
        video_url = url,
        video_title = data['video']['title'],
        video_description = data['video']['description'],
        video_publisher = data['video']['channelTitle'],
        video_tags = data['video']['tags'],
    )
    db.add(db_interaction)
    db.commit()
    db.refresh(db_interaction)
    print('saved base analytics successfully')


@app.get("/analytics/", response_model=List[AnalyticsResponse])
def get_analytics(
    db: Session = Depends(get_db)
):
    return db.query(Analytics).all()


@app.put("/analytics/{interaction_id}")
def save_full_analytics(
    interaction_id: str,
    updated: AnalyticsUpdate,
    db: Session = Depends(get_db)
):
    db_analytic = db.query(Analytics).filter(Analytics.interaction_id == interaction_id).first()
    if not db_analytic:
        raise HTTPException(status_code=404, detail="Question not found")

    for key, value in updated.model_dump().items():
        setattr(db_analytic, key, value)

    db.commit()
    db.refresh(db_analytic)
    print('interaction updated successfully')


def extract_youtube_video_id(url: str) -> str | None:
    """
    Extract YouTube video ID from a URL.
    Supports:
      - https://www.youtube.com/watch?v=VIDEO_ID
      - https://youtu.be/VIDEO_ID
      - https://www.youtube.com/shorts/VIDEO_ID
      - https://www.youtube.com/embed/VIDEO_ID
      - https://www.youtube.com/v/VIDEO_ID
    """

    try:
        parsed = urlparse(url)

        # youtu.be/<id>
        if parsed.netloc in {"youtu.be"}:
            return parsed.path.lstrip("/")

        # youtube.com/*
        if "youtube.com" in parsed.netloc:
            path_parts = parsed.path.split("/")

            # /watch?v=<id>
            if parsed.path == "/watch":
                return parse_qs(parsed.query).get("v", [None])[0]

            # /shorts/<id>
            if len(path_parts) > 2 and path_parts[1] == "shorts":
                return path_parts[2]

            # /embed/<id> or /v/<id>
            if len(path_parts) > 2 and path_parts[1] in {"embed", "v"}:
                return path_parts[2]

    except Exception:
        pass

    return None


@app.get("/aig_tags")
async def get_aigc_tag(
    user_id: str = Query(..., description="Unique user identifier"),
    video_url: str = Query(..., description="YouTube video ID"),
    interaction_id: str = Query(..., description="Type of interaction"),
    db: Session = Depends(get_db)
):
    video_id = extract_youtube_video_id(video_url)
    print(f'user_id: {user_id}\nvideo_id: {video_id}\ninteraction_id: {interaction_id}')
    data = parse_video_data(video_id, yt_key)
    create_base_interaction(data, user_id, video_url, interaction_id, db)
    res = await call_gpts_concurrently(data, creator_prompt, comment_prompt, client)
    return res

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)