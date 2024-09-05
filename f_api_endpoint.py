from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from e_deploy_model_function import deploy_model

app = FastAPI()

# Define the request model for the data input
class PredictRequest(BaseModel):
    model_choice: str = "RF"
    data: list
    verbose: bool = False
    use_pca: bool = True

@app.post("/predict")
async def predict(request: PredictRequest):
    try:
        # Convert list data to numpy array for compatibility
        input_data = np.array(request.data)

        # Handle both single row and multiple rows
        if input_data.ndim == 1:
            input_data = input_data.reshape(1, -1)
        elif input_data.ndim != 2:
            raise ValueError("Data should be 1D (single row) or 2D (multiple rows)")

        # Deploy the model using the imported deploy_model function
        predictions = deploy_model(
            model_choice=request.model_choice,
            data=input_data,
            verbose=request.verbose,
            use_pca=request.use_pca
        )

        return {"predictions": predictions.tolist()}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Run the application using Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)