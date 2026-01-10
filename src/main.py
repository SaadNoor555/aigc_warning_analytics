# Import necessary modules and classes
from fastapi import FastAPI, Depends, HTTPException, Query, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, JSON, Boolean, DateTime, ForeignKey
import sqlalchemy
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.sql import func
from pydantic import BaseModel
from typing import List
import json
from api_handler import call_gpts_concurrently, parse_video_data, src_map
from openai import AsyncOpenAI
from urllib.parse import urlparse, parse_qs
from datetime import datetime, timezone, time, timedelta
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
with open('data/base_prompt_risk.txt', 'r', encoding='utf-8') as pf:
    risk_prompt = pf.read()
with open('data/base_prompt_action.txt', 'r', encoding='utf-8') as pf:
    act_prompt = pf.read()


tactic_map = {
    'impersonation': 'Assume the identity of a real person and take actions on their behalf',
    'appropriated likeness': "Use or alter a person's likeness or other identifying features",
    'sockpuppeting': 'Create synthetic online personas or accounts',
    'non-consensual intimate imagery (ncii)': "Create sexual explicit material using an adult person’s likeness",
    "child sexual abuse material (csam)": "Create child sexual explicit material",
    "falsification": "Fabricate or falsely represent evidence, incl. reports, IDs, documents",
    "intellectual property (ip) infringement": "Use a person's IP without their permission",
    "counterfeit": "Reproduce or imitate an original work, brand or style and pass as real",
    "scaling & amplification": "Automate, amplify, or scale workflows",
    "targeting & personalisation": "Refine outputs to target individuals with tailored attacks",
    "unknown": ""
}





client = AsyncOpenAI(api_key=gpt_key)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*"
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)



# Database setup
DATABASE_URL = "sqlite:///./test.db"
    
response_router = APIRouter(prefix="/survey-responses", tags=["Survey Responses"])
questions_router = APIRouter(prefix="/questions", tags=["Questions"])
analytics_router = APIRouter(prefix="/analytics", tags=["Analytics"])
diary_router = APIRouter(prefix="/user-diary", tags=["User Diary"])



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


class SurveyResponse(Base):
    __tablename__ = "SurveyResponses"

    response_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, nullable=False)
    time_answered = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable= False)
    question_id = Column(Integer, ForeignKey("SurveyQuestions.question_id"), nullable=False)
    answer = Column(String, nullable=False)

    question = relationship("Questions")


class UserDiary(Base):
    __tablename__ = "UserDiary"

    diary_id = Column(Integer, primary_key=True, index=True)
    text = Column(String, nullable=False)
    time_written = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable= False)
    user_id = Column(String, nullable=False)

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
    question_id: int
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

class SurveyResponseCreate(BaseModel):
    user_id: str
    question_id: int
    answer: str

class SurveyResponseUpdate(BaseModel):
    answer: str

class SurveyResponseOut(BaseModel):
    response_id: int
    user_id: str
    time_answered: datetime
    question_id: int
    answer: str

    class Config:
        from_attributes = True

class BulkSurveyAnswer(BaseModel):
    question_id: int
    answer: str

class BulkSurveyResponseCreate(BaseModel):
    user_id: str
    time_answered: datetime
    responses: List[BulkSurveyAnswer]

class UserDiaryCreate(BaseModel):
    user_id: str
    text: str

class UserDiaryResponse(BaseModel):
    diary_id: int
    time_written: datetime
    user_id: str
    text: str

    class Config:
        from_attributes = True


def validate_user_diary_constraints(
    db: Session,
    user_id: str,
    max_entries: int = 10
):
    now = datetime.now(timezone.utc)
    start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    end_of_day = start_of_day + timedelta(days=1)

    # 1️⃣ Same-day check
    same_day_entry = (
        db.query(UserDiary)
        .filter(
            UserDiary.user_id == user_id,
            UserDiary.time_written >= start_of_day,
            UserDiary.time_written < end_of_day
        )
        .first()
    )

    if same_day_entry:
        print('alrady put a response today')
        return False

    # 2️⃣ Max entries check
    total_entries = (
        db.query(func.count(UserDiary.diary_id))
        .filter(UserDiary.user_id == user_id)
        .scalar()
    )

    if total_entries >= max_entries:
        print('user already has max number of entires')
        return False
    
    return True



@diary_router.get(
    "/",
    response_model=List[UserDiaryResponse]
)
def get_all_user_diaries(
    db: Session = Depends(get_db)
):
    return (
        db.query(UserDiary)
        .order_by(UserDiary.time_written.desc())
        .all()
    )



@diary_router.get(
    "/{user_id}",
    response_model=List[UserDiaryResponse]
)
def get_user_diaries(user_id: str, db: Session = Depends(get_db)):
    return (
        db.query(UserDiary)
        .filter(UserDiary.user_id == user_id)
        .order_by(UserDiary.time_written.desc())
        .all()
    )



@diary_router.post(
    "/",
    response_model=UserDiaryResponse
)
def create_user_diary(
    diary: UserDiaryCreate,
    db: Session = Depends(get_db)
):
    # ✅ Reusable validation
    if not validate_user_diary_constraints(db, diary.user_id):
         raise HTTPException(
            status_code=400,
            detail="User has reached the maximum number of diary entries today or for the entirity of the study."
        )

    new_diary = UserDiary(user_id=diary.user_id, text=diary.text)
    db.add(new_diary)
    db.commit()
    db.refresh(new_diary)

    return new_diary



@diary_router.delete(
    "/{diary_id}",
    status_code=204
)
def delete_user_diary(
    diary_id: int,
    db: Session = Depends(get_db)
):
    diary = (
        db.query(UserDiary)
        .filter(UserDiary.diary_id == diary_id)
        .first()
    )

    if not diary:
        raise HTTPException(
            status_code=404,
            detail="Diary entry not found."
        )

    db.delete(diary)
    db.commit()

@questions_router.post("/", response_model=QuestionsResponse)
def create_question(
    question: QuestionsCreate,
    db: Session = Depends(get_db)
):
    db_question = Questions(**question.model_dump())
    print(db_question)
    db.add(db_question)
    db.commit()
    db.refresh(db_question)
    return db_question


@questions_router.get("/", response_model=List[QuestionsResponse])
def get_questions(
    db: Session = Depends(get_db)
):
    return db.query(Questions).all()


@questions_router.get("/{question_id}", response_model=QuestionsResponse)
def get_question(
    question_id: int,
    db: Session = Depends(get_db)
):
    question = db.query(Questions).filter(Questions.question_id == question_id).first()
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")
    return question

def day_bounds(dt: datetime):
    start = datetime.combine(dt.date(), time.min)
    end = datetime.combine(dt.date(), time.max)
    return start, end


@questions_router.get(
    "/unanswered/{user_id}",
    response_model=List[QuestionsResponse]
)
def get_unanswered_questions_today(
    user_id: str,
    db: Session = Depends(get_db)
):
    # Use naive UTC to match stored values
    now = datetime.utcnow()
    start, end = day_bounds(now)

    # 1️⃣ Fetch question_ids answered today
    answered_question_ids = [
        qid for (qid,) in (
            db.query(SurveyResponse.question_id)
            .filter(
                SurveyResponse.user_id == user_id,
                SurveyResponse.time_answered >= start,
                SurveyResponse.time_answered <= end,
            )
            .all()
        )
    ]

    # 2️⃣ If user answered nothing today → return all questions
    if not answered_question_ids:
        print('no ans')
        return db.query(Questions).all()

    # 3️⃣ Fetch unanswered questions
    unanswered_questions = (
        db.query(Questions)
        .filter(~Questions.question_id.in_(answered_question_ids))
        .all()
    )

    return unanswered_questions


@questions_router.put("/{question_id}", response_model=QuestionsResponse)
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


@questions_router.delete("/{question_id}")
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


@analytics_router.get("/", response_model=List[AnalyticsResponse])
def get_analytics(
    db: Session = Depends(get_db)
):
    return db.query(Analytics).all()


@analytics_router.put("/{interaction_id}")
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

def day_bounds(dt: datetime):
    start = datetime.combine(dt.date(), time.min)
    end = datetime.combine(dt.date(), time.max)
    return start, end

def validate_survey_responses(
    *,
    db: Session,
    user_id: str,
    time_answered: datetime,
    responses: list[tuple[int, str]]
):
    """
    responses: List of (question_id, answer)
    """

    # 1️⃣ Check duplicate question_ids in request
    question_ids = [qid for qid, _ in responses]
    if len(question_ids) != len(set(question_ids)):
        raise HTTPException(
            status_code=400,
            detail="Duplicate question_id in request"
        )

    # 2️⃣ Check existing answers on same day
    start, end = day_bounds(time_answered)

    existing = (
        db.query(SurveyResponse.question_id)
        .filter(
            SurveyResponse.user_id == user_id,
            SurveyResponse.question_id.in_(question_ids),
            SurveyResponse.time_answered.between(start, end)
        )
        .all()
    )

    if existing:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "User already answered some questions today",
                "question_ids": [q[0] for q in existing]
            }
        )

@response_router.post("/", response_model=SurveyResponseOut)
def create_response(
    payload: SurveyResponseCreate,
    db: Session = Depends(get_db)
):
    now = datetime.utcnow()

    validate_survey_responses(
        db=db,
        user_id=payload.user_id,
        time_answered=now,
        responses=[(payload.question_id, payload.answer)]
    )

    response = SurveyResponse(
        user_id=payload.user_id,
        question_id=payload.question_id,
        answer=payload.answer,
        time_answered=now
    )

    db.add(response)
    db.commit()
    db.refresh(response)
    return response

@response_router.get("/", response_model=List[SurveyResponseOut])
def get_all_responses(db: Session = Depends(get_db)):
    return db.query(SurveyResponse).all()


@response_router.get("/question/{question_id}", response_model=List[SurveyResponseOut])
def get_responses_by_question(
    question_id: int,
    db: Session = Depends(get_db)
):
    return db.query(SurveyResponse).filter_by(question_id=question_id).all()


@response_router.delete("/{response_id}")
def delete_response(
    response_id: int,
    db: Session = Depends(get_db)
):
    response = db.query(SurveyResponse).filter_by(response_id=response_id).first()
    if not response:
        raise HTTPException(status_code=404, detail="Response not found")

    db.delete(response)
    db.commit()
    return {"detail": "Response deleted"}


@response_router.get("/user/{user_id}", response_model=List[SurveyResponseOut])
def get_user_responses(
    user_id: str,
    db: Session = Depends(get_db)
):
    return db.query(SurveyResponse).filter_by(user_id=user_id).all()

@response_router.post("/bulk")
def create_bulk_responses(
    payload: BulkSurveyResponseCreate,
    db: Session = Depends(get_db)
):
    responses = [
        (r.question_id, r.answer)
        for r in payload.responses
    ]

    validate_survey_responses(
        db=db,
        user_id=payload.user_id,
        time_answered=payload.time_answered,
        responses=responses
    )

    objects = []

    for r in payload.responses:
        if r.answer=='':
            continue
        objects.append(
            SurveyResponse(
                    user_id=payload.user_id,
                    time_answered=payload.time_answered,
                    question_id=r.question_id,
                    answer=r.answer
                )
        )
        
    print(len(objects))

    db.bulk_save_objects(objects)
    db.commit()

    return {"inserted": len(objects)}

def createPlatformLabel(platform_label: str):
    response = ''
    if len(platform_label)>2:
        response += \
f'Based on YouTube’s label, this video appears to be Al Generated content. YouTube marks it as {platform_label} for transparency. Sound or visuals were significantly edited or digitally generated by AI.'
    else:
        response += \
f'Based on YouTube’s label, there is no indication whether this video contains AI Generated content or not.'
    return response


def createCommentsLabel(commentRes):
    try:
        tot = len(commentRes.keys())
        if tot==0:
            return 'This video does not have any comments hence could not analyze for AI Generated content indications'
        ys = 0
        for key in commentRes.keys():
            if commentRes[key].lower()=='yes':
                ys+=1
        ys_per = (ys/tot) * 100
        response = f'From total comments, {round(ys_per, 2)}% indicate this video might be generated or altered with AI.'
        return response
    except:
        return 'Could not analyze comments for AI Generated content indications'

def createCreatorLabel(creator):
    response = ''
    if creator=='yes':
        response = 'Based on the creator’s own indications, such as title, description, and hashtags, the creator of this video indicates it contains AI-generated content.'
    else:
        response = 'Based on the creator’s own indications, such as title, description, and hastags, the creator of this video does not indicate whether it contains AI-generated content or not.'
    return response

def createRiskEvalLablel(risk_eval):
    print(risk_eval)
    risk_tactic = risk_eval['tactic'] if 'tactic' in risk_eval else 'unknown'
    return {
        "level": risk_eval['risk'] if 'risk' in risk_eval else 'unknown',
        "category": risk_eval['category'] if 'category' in risk_eval else 'unknown',
        "tactics_name": risk_eval['tactic'] if 'tactic' in risk_eval else 'unknown',
        "tactics_details": tactic_map.get(risk_tactic.lower())
    }

def createRecommendations(sources_obj):
    sources = []
    try:
        sources = sources_obj['sources']
    except:
        pass
    sources_tag = ' :'
    for src in sources:
        sources_tag += f'<a href={src_map[src.lower()]}>{src}</a>, '
    sources_tag = sources_tag[:-2]
    return {
        "before_share": "Sharing AI-generated content without disclosure can cause others to misinterpret it as authentic and spread it further. It may lead to scams in the worst cases, as misleading medical content can be used to trick viewers into false beliefs or fraudulent medical purchases. Please indicate in your post or text that the content is AI-generated.",
        "check_sources": f"Do not take the content at face value. Search for corroboration in trusted sources: {sources_tag if len(sources_tag)>2 else "<a href=google.com>Google</a>"}."
    }

@app.get("/aig_tags")
async def get_aigc_tag(
    user_id: str = Query(..., description="Unique user identifier"),
    video_url: str = Query(..., description="YouTube video ID"),
    interaction_id: str = Query(..., description="Type of interaction"),
    platform_label: str = Query(..., description="Platform label provided"),
    db: Session = Depends(get_db)
):
    video_id = extract_youtube_video_id(video_url)
    print(f'user_id: {user_id}\nvideo_id: {video_id}\ninteraction_id: {interaction_id}')
    data = parse_video_data(video_id, yt_key)
    create_base_interaction(data, user_id, video_url, interaction_id, db)
    res = await call_gpts_concurrently(data, creator_prompt, comment_prompt, risk_prompt, act_prompt, client)
    # res['diary'] = validate_user_diary_constraints(db, user_id)

    if res['creator_tag'] != 'yes' and len(platform_label)<3:
        return {'payload': {'show': False}}
    print(res)
    payload = {
        'show': True,
        'platform_indication': createPlatformLabel(platform_label),
        'creator_metadata': createCreatorLabel(res['creator_tag']),
        'community_feedback': createCommentsLabel(res['community_tags']),
        'risk_evaluation': createRiskEvalLablel(res['risk_evaluation']),
        'recommendation': createRecommendations(res['sources']),
        'diary': validate_user_diary_constraints(db, user_id)
    }
    print(payload)
    return {"payload":payload}

app.include_router(response_router)
app.include_router(questions_router)
app.include_router(analytics_router)
app.include_router(diary_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


# sudo systemctl stop fastapi
# sudo systemctl start fastapi
