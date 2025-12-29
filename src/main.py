# Import necessary modules and classes
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy import create_engine, Column, Integer
import sqlalchemy
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel

# FastAPI app instance
app = FastAPI()

# Database setup
DATABASE_URL = "sqlite:///./test.db"


engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = sqlalchemy.orm.declarative_base()

# Database model
class OpenDuration(Base):
    __tablename__ = "OpenDurations"
    id = Column(Integer, primary_key=True, index=True)
    time = Column(Integer)


# Create tables
Base.metadata.create_all(bind=engine)

# Dependency to get the database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Pydantic model for request data
class DurationCreate(BaseModel):
    time: int

class DurationResponse(BaseModel):
    id: int
    time: int


# API endpoint to create an item
@app.post("/durations/", response_model=DurationResponse)
async def create_item(item: DurationResponse, db: Session = Depends(get_db)):
    db_item = OpenDuration(**item.model_dump())
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item

# API endpoint to read an item by ID
@app.get("/items/{item_id}", response_model=DurationResponse)
async def read_item(item_id: int, db: Session = Depends(get_db)):
    db_item = db.query(OpenDuration).filter(OpenDuration.id == item_id).first()
    if db_item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return db_item


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)