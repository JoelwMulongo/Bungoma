{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "047703c6",
   "metadata": {
    "_cell_guid": "b1076dfc-b9ad-4769-8c92-a6c4dae69d19",
    "_uuid": "8f2839f25d086af736a60e9eeb907d3b93b6e0e5",
    "execution": {
     "iopub.execute_input": "2025-05-17T08:59:43.308158Z",
     "iopub.status.busy": "2025-05-17T08:59:43.307329Z",
     "iopub.status.idle": "2025-05-17T08:59:47.754292Z",
     "shell.execute_reply": "2025-05-17T08:59:47.753460Z"
    },
    "papermill": {
     "duration": 4.452486,
     "end_time": "2025-05-17T08:59:47.755975",
     "exception": false,
     "start_time": "2025-05-17T08:59:43.303489",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Import libraries\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV\n",
    "from sklearn.preprocessing import LabelEncoder, StandardScaler\n",
    "from sklearn.ensemble import RandomForestRegressor\n",
    "from sklearn.metrics import mean_squared_error, r2_score\n",
    "from sklearn.pipeline import Pipeline\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "# Set plot style\n",
    "plt.style.use('seaborn')\n",
    "%matplotlib inline"
   ]
  }
 ],
 "metadata": {
  "kaggle": {
   "accelerator": "none",
   "dataSources": [
    {
     "datasetId": 7442387,
     "sourceId": 11845190,
     "sourceType": "datasetVersion"
    }
   ],
   "dockerImageVersionId": 31040,
   "isGpuEnabled": false,
   "isInternetEnabled": true,
   "language": "python",
   "sourceType": "notebook"
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.11"
  },
  "papermill": {
   "default_parameters": {},
   "duration": 9.859371,
   "end_time": "2025-05-17T08:59:48.476516",
   "environment_variables": {},
   "exception": null,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2025-05-17T08:59:38.617145",
   "version": "2.6.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
