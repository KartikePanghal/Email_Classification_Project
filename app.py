from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError
from utils import mask_pii, demask_pii
from models import classify_email
import logging

# Initialize FastAPI app
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Request model
class EmailRequest(BaseModel):
    email_body: str

# Root route
@app.get("/")
def read_root():
    return {"message": "Welcome to the Email Classification API"}

# Favicon route
@app.get("/favicon.ico")
def favicon():
    return JSONResponse(content={}, status_code=204)

@app.post("/classify_email")
def classify_email_api(request: EmailRequest):
    try:
        email_body = request.email_body
        logger.info("Received email for classification.")

        # Mask PII
        masked_email, masked_entities = mask_pii(email_body)
        logger.info("PII masking completed.")

        # Classify email
        category = classify_email(masked_email)
        logger.info(f"Email classified as: {category}")

        # Demask PII
        demasked_email = demask_pii(masked_email, masked_entities)
        logger.info("PII demasking completed.")

        # Return response
        return {
            "input_email_body": email_body,
            "list_of_masked_entities": masked_entities,
            "masked_email": masked_email,
            "category_of_the_email": category
        }

    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail="Invalid input format.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")
