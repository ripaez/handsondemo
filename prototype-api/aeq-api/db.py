from sqlalchemy import create_engine, TypeDecorator, Column, Integer, String, DateTime, BLOB, ForeignKey, CHAR, Boolean, \
    BigInteger
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.sql import func
import uuid

SQLALCHEMY_DATABASE_URL = 'sqlite:///./aequitas.sqlite'

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class _Base:
    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


Base = declarative_base(cls=_Base)


class Project(Base):
    __tablename__ = "Project"

    id = Column(String(36), primary_key=True, nullable=False)
    props = Column(String(), nullable=True)


class ProjectData(Base):
    __tablename__ = "ProjectDataBlock"
    pid = Column(String(36), ForeignKey('Project.id'), primary_key=True)
    key = Column(String(128), primary_key=True)  # node identifier - ex: the dataset
    skey = Column(String(128), nullable=True, primary_key=True, default="base")  # version identifier - ex: after rebalancing
    parent = Column(String(128), nullable=True)  # parent skey - ex: base - null means base
    data = Column(String(), nullable=False)
    dt = Column(DateTime(timezone=True), onupdate=func.now())

Base.metadata.create_all(bind=engine)