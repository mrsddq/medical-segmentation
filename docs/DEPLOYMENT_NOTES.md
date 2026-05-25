# Deployment Notes

## Demo Target

Use a de-identified sample image in a Streamlit app to show:

- input slice
- predicted mask
- overlay
- threshold slider

## Serving Path

1. Save model checkpoint and config together.
2. Add preprocessing parity checks.
3. Package inference behind FastAPI or Streamlit.
4. Keep patient data and model weights out of Git.
5. Log only de-identified artifacts.
