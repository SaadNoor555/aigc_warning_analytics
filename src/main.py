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
from api_handler import call_gpts_concurrently, parse_video_data, src_map, get_responses_today
from openai import AsyncOpenAI
from urllib.parse import urlparse, parse_qs
from datetime import datetime, timezone, time, timedelta
from google.oauth2.service_account import Credentials
import gspread
import csv
import io
from fastapi.responses import StreamingResponse
# FastAPI app instance

# TODO: RESPONSE TYPES AND CODES

with open('data/config.json') as jf:
    keys = json.load(jf)

yt_key = keys['YT_API_KEY']
gpt_key = keys['GPT_API_KEY']
sheet_id = keys['SHEET_ID']
worksheet = keys['WORKSHEET']
server_pass = keys['SERVER_PASS']


SERVICE_ACCOUNT_FILE = "data/service_account.json"

scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
creds = Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=scopes
)
# print(creds.service_account_email)
client = gspread.authorize(creds)

sheet = client.open_by_key(sheet_id)
worksheet = sheet.worksheet(worksheet)

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
    popup               = Column(Boolean, nullable=False)
    platform_label            = Column(String, nullable = True)
    total_time_spent          = Column(Integer, nullable=True)
    time_spent_platform_label = Column(Integer, nullable=True)
    time_spent_creator_label  = Column(Integer, nullable=True)
    time_spent_comment_label  = Column(Integer, nullable=True)
    time_spent_risk_label     = Column(Integer, nullable=True)
    time_spent_recommendation = Column(Integer, nullable=True)
    audio_warning             = Column(Boolean, nullable=True)
    survey                    = Column(Boolean, nullable=True)
    partial                   = Column(Boolean, nullable=True)
    platform_indication       = Column(String, nullable=True)
    creator_indication        = Column(String, nullable=True)
    comment_indication        = Column(String, nullable=True)
    risk_level                = Column(String, nullable=True)
    content_category          = Column(String, nullable=True)
    ai_tactic                 = Column(String, nullable=True)
    action_before_share       = Column(String, nullable=True)
    action_trusted_sources    = Column(String, nullable=True)


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


class ResponseSave(BaseModel):
    popup : bool
    platform_indication : str | None = None
    creator_indication : str | None = None
    comment_indication : str | None = None
    risk_level : str | None = None
    content_category: str | None = None
    ai_tactic : str | None = None
    action_before_share : str | None = None
    action_trusted_sources : str | None = None
    

class AnalyticsUpdate(BaseModel):
    total_time_spent : int | None
    time_spent_platform_label : int | None
    time_spent_creator_label : int | None
    time_spent_comment_label : int | None
    time_spent_risk_label : int | None
    time_spent_recommendation : int | None
    audio_warning: bool | None
    partial: bool | None


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
    popup           : bool
    total_time_spent : int | None
    time_spent_platform_label : int | None
    time_spent_creator_label : int | None
    time_spent_comment_label : int | None
    time_spent_risk_label : int | None
    time_spent_recommendation : int | None
    audio_warning: bool | None
    survey: bool | None
    partial: bool | None
    platform_indication : str | None = None
    creator_indication : str | None = None
    comment_indication : str | None = None
    risk_level : str | None = None
    content_category: str | None = None
    ai_tactic : str | None = None
    action_before_share : str | None = None
    action_trusted_sources : str | None = None

    class Config:
        from_attributes = True  # required for SQLAlchemy → Pydantic


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

class TakenSurvey(BaseModel):
    survey: bool



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


@diary_router.get(
    "/download/csv"
)
def download_user_diary_csv(
    db: Session = Depends(get_db),
    password: str = Query(description="Enter server password"),
    # Optional filters
    start_time: datetime | None = Query(None, description="Filter: time_written >= start_time (UTC)"),
    end_time: datetime | None = Query(None, description="Filter: time_written <= end_time (UTC)"),
    user_id: str | None = Query(None, description="Filter by user_id"),
):
    if password != server_pass:
        raise HTTPException(
            status_code=404,
            detail="Incorrect password."
            )

    query = db.query(UserDiary)

    # Apply filters
    if start_time:
        query = query.filter(UserDiary.time_written >= start_time)

    if end_time:
        query = query.filter(UserDiary.time_written <= end_time)

    if user_id:
        query = query.filter(UserDiary.user_id == user_id)

    results = query.order_by(UserDiary.time_written.asc()).all()

    # CSV buffer
    output = io.StringIO()
    writer = csv.writer(output)

    # Header row
    writer.writerow([
        "diary_id",
        "time_written",
        "user_id",
        "text",
    ])

    # Data rows
    for row in results:
        writer.writerow([
            row.diary_id,
            row.time_written.isoformat() if row.time_written else None,
            row.user_id,
            row.text,
        ])

    output.seek(0)

    filename = f"user_diary_export_{datetime.utcnow().isoformat()}.csv"

    return StreamingResponse(
        output,
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        }
    )

@questions_router.get(
    "/unanswered/{user_id}",
    response_model=TakenSurvey
)
def get_unanswered_questions_today(
    user_id: str,
    db: Session = Depends(get_db)
):
    res = get_responses_today(user_id, worksheet)
    # print(res)
    return TakenSurvey(survey=(not res))


def create_base_interaction(data: dict, user_id: str, url: str, interaction_id: str, platform_label: str, survey: bool, db: Session):
    db_interaction = Analytics(
        user_id = user_id,
        interaction_id = interaction_id,
        video_url = url,
        video_title = data['video']['title'],
        video_description = data['video']['description'],
        video_publisher = data['video']['channelTitle'],
        video_tags = data['video']['tags'],
        platform_label = platform_label,
        survey = survey,
        popup = False
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
        raise HTTPException(status_code=404, detail="analytics not found")

    for key, value in updated.model_dump().items():
        setattr(db_analytic, key, value)

    db.commit()
    db.refresh(db_analytic)
    print('interaction updated successfully')

def save_gpt_responses(interaction_id: str, respone: ResponseSave, db: Session):
    db_analytic = db.query(Analytics).filter(Analytics.interaction_id == interaction_id).first()
    if not db_analytic:
        print(f'analytic {interaction_id} not found')
    for key, value in respone.model_dump().items():
        setattr(db_analytic, key, value)
    
    db.commit()
    db.refresh(db_analytic)
    print('interaction updated with responses successfully')


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
        if ys>0:
            response = f'Based on the total comments, {round(ys_per, 2)}% indicate this video might be generated or altered by AI.'
        else:
            response = f'Based on the total comments there is no indication that this content might be generated or altered by AI'
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
    # print(risk_eval)
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
        sources_tag += f'<a style="color:#887aad" href={src_map[src.lower()]}>{src}</a>, '
    sources_tag = sources_tag[:-2]
    return {
        "before_share": "Sharing AI-generated content without disclosure can cause others to misinterpret it as authentic and spread it further. It may lead to scams in the worst cases, as misleading medical content can be used to trick viewers into false beliefs or fraudulent medical purchases. Please indicate in your post or text that the content is AI-generated.",
        "check_sources": f"Do not take the content at face value. Search for corroboration in trusted sources: {sources_tag if len(sources_tag)>2 else "<a href=google.com>Google</a>"}."
    }


@analytics_router.get("/download/csv")
def download_analytics_csv(
    db: Session = Depends(get_db),
    password: str = Query(description="Enter server password"),
    # Optional filters
    start_time: datetime | None = Query(None, description="Filter: time_requested >= start_time (UTC)"),
    end_time: datetime | None = Query(None, description="Filter: time_requested <= end_time (UTC)"),
    user_id: str | None = Query(None, description="Filter by user_id"),
    popup: bool | None = Query(None, description="Filter by popup true/false"),
):
    if password != server_pass:
        raise HTTPException(
            status_code=404,
            detail="Incorrect password."
            )

    query = db.query(Analytics)

    # Apply filters
    if start_time:
        query = query.filter(Analytics.time_requested >= start_time)

    if end_time:
        query = query.filter(Analytics.time_requested <= end_time)

    if user_id:
        query = query.filter(Analytics.user_id == user_id)

    if popup is not None:
        query = query.filter(Analytics.popup == popup)

    results = query.all()

    # CSV buffer
    output = io.StringIO()
    writer = csv.writer(output)

    # Header row
    writer.writerow([
        "interaction_id",
        "user_id",
        "video_url",
        "video_title",
        "video_description",
        "video_tags",
        "video_publisher",
        "time_requested",
        "popup",
        "platform_label",
        "total_time_spent",
        "time_spent_platform_label",
        "time_spent_creator_label",
        "time_spent_comment_label",
        "time_spent_risk_label",
        "time_spent_recommendation",
        "audio_warning",
        "survey",
        "partial",
        "platform_indication",
        "creator_indication",
        "comment_indication",
        "risk_level",
        "content_category",
        "ai_tactic",
        "action_before_share",
        "action_trusted_sources",
    ])

    # Data rows
    for row in results:
        total_time = row.total_time_spent if (row.total_time_spent and row.total_time_spent<3600*2*1000) else 0
        platform_time = row.time_spent_platform_label if (row.time_spent_platform_label and row.time_spent_platform_label<total_time) else 0
        creator_time = row.time_spent_creator_label if (row.time_spent_creator_label and row.time_spent_creator_label<total_time) else 0
        comments_time = row.time_spent_comment_label if (row.time_spent_comment_label and row.time_spent_comment_label<total_time) else 0
        risk_time = row.time_spent_risk_label if (row.time_spent_risk_label and row.time_spent_risk_label<total_time) else 0
        rec_time = row.time_spent_recommendation if (row.time_spent_recommendation and row.time_spent_recommendation<total_time) else 0

        writer.writerow([
            row.interaction_id,
            row.user_id,
            row.video_url,
            row.video_title,
            row.video_description,
            ",".join(row.video_tags) if row.video_tags else None,
            row.video_publisher,
            row.time_requested.isoformat() if row.time_requested else None,
            row.popup,
            row.platform_label,
            total_time,
            platform_time,
            creator_time,
            comments_time,
            risk_time,
            rec_time,
            row.audio_warning,
            row.survey,
            row.partial,
            row.platform_indication,
            row.creator_indication,
            row.comment_indication,
            row.risk_level,
            row.content_category,
            row.ai_tactic,
            row.action_before_share,
            row.action_trusted_sources,
        ])

    output.seek(0)

    filename = f"analytics_export_{datetime.utcnow().isoformat()}.csv"

    return StreamingResponse(
        output,
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        }
    )

@app.get("/aig_tags")
async def get_aigc_tag(
    user_id: str = Query(..., description="Unique user identifier"),
    video_url: str = Query(..., description="YouTube video ID"),
    interaction_id: str = Query(..., description="Type of interaction"),
    platform_label: str = Query(..., description="Platform label provided"),
    db: Session = Depends(get_db)
):
    video_id = extract_youtube_video_id(video_url)
    if video_id==None:
        print(f'not youtube video: {video_url}')
        return {'payload': {'show': False}}
    # print(f'user_id: {user_id}\nvideo_id: {video_id}\ninteraction_id: {interaction_id}')
    data = parse_video_data(video_id, yt_key)
    survey = (not get_responses_today(user_id, worksheet))
    create_base_interaction(data, user_id, video_url, interaction_id, platform_label, survey, db)
    res = await call_gpts_concurrently(data, creator_prompt, comment_prompt, risk_prompt, act_prompt, client)

    response = ResponseSave(popup=False)

    if res['creator_tag'] != 'yes' and len(platform_label)<3:
        save_gpt_responses(interaction_id, response, db)
        return {'payload': {'show': False}}
    # print(res)
    response.popup = True
    payload = {
        'show': True,
        'platform_indication': createPlatformLabel(platform_label),
        'creator_metadata': createCreatorLabel(res['creator_tag']),
        'community_feedback': createCommentsLabel(res['community_tags']),
        'risk_evaluation': createRiskEvalLablel(res['risk_evaluation']),
        'recommendation': createRecommendations(res['sources']),
        'diary': True,
        'survey': survey
    }

    response.platform_indication = payload['platform_indication']
    response.creator_indication = payload['creator_metadata']
    response.comment_indication = payload['community_feedback']
    response.risk_level = payload['risk_evaluation']['level']
    response.content_category = payload['risk_evaluation']['category']
    response.ai_tactic = payload['risk_evaluation']['tactics_name']
    response.action_before_share = payload['recommendation']['before_share']
    response.action_trusted_sources = payload['recommendation']['check_sources']
    save_gpt_responses(interaction_id, response, db)

    # print(response)
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
