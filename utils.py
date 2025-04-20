import re
import logging

# Configure logging
logger = logging.getLogger(__name__)

def mask_pii(email_body):
    try:
        masked_entities = []
        patterns = {
            "full_name": r"\b[A-Z][a-z]+\s[A-Z][a-z]+\b",
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "phone_number": r"\b\d{10}\b",
            "dob": r"\b\d{2}/\d{2}/\d{4}\b",
            "aadhar_num": r"\b\d{12}\b",
            "credit_debit_no": r"\b\d{16}\b",
            "cvv_no": r"\b\d{3}\b",
            "expiry_no": r"\b\d{2}/\d{2}\b"
        }

        for entity, pattern in patterns.items():
            matches = re.finditer(pattern, email_body)
            for match in matches:
                start, end = match.span()
                original_value = match.group()
                masked_entities.append({
                    "position": [start, end],
                    "classification": entity,
                    "entity": original_value
                })
                email_body = email_body[:start] + f"[{entity}]" + email_body[end:]

        logger.info("PII masking completed successfully.")
        return email_body, masked_entities

    except Exception as e:
        logger.error(f"Error during PII masking: {e}")
        raise

def demask_pii(masked_email, masked_entities):
    try:
        for entity in masked_entities:
            start, end = entity["position"]
            original_value = entity["entity"]
            masked_email = masked_email.replace(f"[{entity['classification']}]", original_value, 1)

        logger.info("PII demasking completed successfully.")
        return masked_email

    except Exception as e:
        logger.error(f"Error during PII demasking: {e}")
        raise
