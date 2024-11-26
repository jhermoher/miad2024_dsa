# setup.py
from setuptools import setup, find_packages

setup(
    name="news_categorization_pkg",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi==0.115.5",
        "uvicorn==0.32.1",
        "pydantic==2.10.0",
        "pydantic-settings==2.6.1",
        "nltk==3.9.1",
        "scikit-learn==1.5.2",
        "numpy==2.1.3",
        "pandas==2.2.3",
        "joblib==1.4.2",
        "vaderSentiment==3.3.2",
        "python-dotenv==1.0.1",
        "httpx>=0.25.2"
    ],
    extras_require={
        'test': [
            'pytest==8.3.3',
            'httpx>=0.25.2'
        ],
        'dev': [
            'tqdm==4.67.0'
        ]
    },
    entry_points={
        'console_scripts': [
            'news-api=app.main:start_server',
        ],
    },
    python_requires=">=3.8",
    description="News categorization and sentiment analysis API",
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Operating System :: OS Independent",
    ],
)