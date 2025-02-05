"""Importing required libraries to 
clean and pre-process the dataset and 
training the model with data"""
import pandas as pd
from sklearn import linear_model
from sklearn import preprocessing
from matplotlib import pyplot as plt
import numpy as np


"""### Class LoadData:

purpose: 
    1. To take the required datasets, feature columns 
        and label column as argument.
    2. First, checking validation of file and column 
        existance in dataset and then converting into 
        pandas dataframe for handling the dataframe."""
class LoadData:
    # protected class attributes
    _original_df : pd.DataFrame = None # saving the original dataframe from given split range row 
    _data_frame : pd.DataFrame = None # for storing the dataframe here
    _features : list[str] = None # for storing the features columns
    _label : str = None # for storing the label column here
    _split_range : int = None # for splitting the data
    _label_encoding_column : str = None # storing column name to label encode
    _one_hot_encoding_column : list[str] = None # storing list of column for one-hot-encoding
    
    
    """LoadData class's init() constuctor method returns None.
    
    purpose: To take the dataset and features column and label 
             column, split range as argument."""  
    def __init__(self, dataset : str, features : list[str], 
                 label : str, split_range : int, 
                 label_encoding_column : str, 
                 one_hot_encoding_column : list[str]) -> None:
        # for handling errors.
        try:
            # checking file type validation
            if (dataset.endswith(".csv")):
                # converting the dataset into pandas dataframe
                self._data_frame = pd.read_csv(dataset)
                # 
                self._original_df = pd.read_csv(dataset)
                
                """checking all the features are exists in dataframe 
                    or not using for loop."""
                for feature in features:
                    # if feature not exists in dataframe then raising index error
                    if (feature not in self._data_frame.columns):
                        raise IndexError(f"Error! {feature} named column doesn't exist in {dataset} dataset.")
                    
                """checking the label exists in the dataframe or not"""
                if (label not in self._data_frame.columns):
                    # if label not exists in dataframe then raising index error
                    raise IndexError(f"Error! {label} named column doesn't exist in {dataset} dataset.")
                
                """After checking all the validations, 
                    then we will store the dataset."""
                self._features = features # storing the list of feature's columns
                self._original_df = self._original_df.loc[split_range: , :] # storing from split range rows
                self._label = label # storing the label column
                self._split_range = split_range # storing the split range
                self._label_encoding_column = label_encoding_column # storing the column name
                self._one_hot_encoding_column = one_hot_encoding_column # storing the column name
                
                # showing this message for acknowledgement
                print(f"\nDataset {dataset} has loaded successfully in pandas dataframe...")
            
            # if file type not matched, then raising type error
            else:
                raise TypeError("Error! LoadData class only accepts .csv file.")
        
        # for handling IndexError
        except IndexError as i:
            print(f"\nError message from LoadData class's constructor() method: {i}")
        
        # for handling TypeError
        except TypeError as t:
            print(f"\nError message from LoadData class's constructor() method: {t}")
        
        # for handling any other errors
        except Exception as e:
            print(f"\nError message from LoadData class's constructor() method: {e}")



"""### Class CleanAndProcessData inheriting LoadData class:

purpose:
    1. To clean the data by removing duplicate rows.
    2. To removing the null values"""
class CleanAndProcessData(LoadData):
    """** Method data_cleaning():
    
    Purpose: To call fill_null_value() and remove_duplicate_values() method"""
    def data_cleaning(self) -> None:
        """calling the fill_null_values() method 
        for removing null values"""
        self.fill_null_value()
        
        """calling the remove_duplicate_values() method 
        for removing duplicate values"""
        self.remove_duplicate_values()
    
    
    """** Method fill_null_value() returns None:
    
    purpose: To check for any null value, if found then 
             handling in this method."""
    def fill_null_value(self) -> None:
        # for handling the error here
        try:
            # fist checking if any null value found or not
            if (self._data_frame.isnull().values.any()):
                """removing rows with null values, as in the dataset there are
                    not any null values and more than 10000+ rows"""
                self._data_frame.dropna(how = "any", axis = 0, inplace = True)
                
                # showing for acknowledgement
                print("\nNull values are removed succesfully from the dataset.")
            
            # otherwise, showing for an acknowledgement    
            else:
                print("\nWow, the dataset doesn't contain any single null values!")
                
        # showing this error message into screen
        except Exception as e:
            print(f"\nError message from CleanAndProcessData class's fill_null_value() method: {e}")
    
    
    """** Method remove_duplicate_values() returns None:
    
    purpose: To check for any duplicate value, if found then 
             handling in this method."""        
    def remove_duplicate_values(self) -> None:
        # for handling the error here
        try:
            """removing the duplicate values, otherwise they will lead
                wrong label output."""
            if (self._data_frame.duplicated().values.any()):
                # removing duplicate values while keeping the first occurance
                self._data_frame.drop_duplicates(keep = "first", inplace = True)
                
                # showing for acknowledgement
                print("\nDuplicate values are removed succesfully from the dataset.")
                
            # otherwise, showing this for acknowledgement
            else:
                print("\nWow, the dataset doesn't contain any single duplicate values!")
                
        # showing this error message into screen
        except Exception as e:
            print(f"\nError message from CleanAndProcessData class's remove_duplicate_values() method: {e}")


"""### Class FeatureEngineering inherits CleanAndProcessData.

Purpose: To encode the features with corresponding techniques"""
class FeatureEngineering(CleanAndProcessData):  
    # creating protected instance of LabelEncoder class from preprocessing sub module
    _label_encoder = preprocessing.LabelEncoder()
    
    
    """** Method feature_engineer() that returns None.
    
    Purpose: To encode the the column values with ohe (one-hot-encoding) 
            and le (label-encoding) methods"""        
    def feature_engineer(self) -> None:  
        try:  
            # calling one_hot_encoder() method to one-hot-encode the columns
            self.one_hot_encoder()  
            # calling label_encoder() method to label-encode the column
            self.label_encoder()
            
        # showing this error message into screen
        except Exception as e:
            print(f"\nError message from FeatureEngineering class's feature_engineer() method: {e}")
          
    
    """** Method label_encoder() that returns None.
    
    Purpose: To label encode the column."""
    def label_encoder(self) -> None:
        # for handling the error here     
        try:
            # first checking if the column exist or not in the features
            if self._label_encoding_column in self._data_frame.columns:
                # label encoding the that column by using fit_transform() method from LabelEncoder class
                self._data_frame[self._label_encoding_column] = self._label_encoder.fit_transform(
                                                            self._data_frame[self._label_encoding_column])
                
            # otherwise raising index error
            else:
                raise IndexError(f"Error! {self._label_encoding_column} named column doesn't exist.")
            
        # showing this error message into screen
        except IndexError as i:
            print(f"\nError message from FeatureEngineering class's label_encoder() method: {i}")
            
        # showing this error message into screen
        except Exception as e:
            print(f"\nError message from FeatureEngineering class's label_encoder() method: {e}")
            
            
    """** Method one_hot_encoder() that returns None.
    
    Purpose: To one hot encode the columns."""
    def one_hot_encoder(self) -> None:
        # for handling the error here     
        try:
            # first checking if the column exist or not in the features using loop
            for category_column in self._one_hot_encoding_column:
                # checking in features that exist or not
                if category_column in self._data_frame.columns:
                    """creating dummies of that columns by removing the first column 
                    because off multilinear trap"""
                    dummies = pd.get_dummies(self._data_frame[category_column], 
                                             dtype = int, drop_first = True)
                    
                    # droping that category column from the features
                    self._data_frame.drop([category_column], axis = 1, inplace = True)
                    
                    # concating the dummy column and updating orinal features column
                    self._data_frame = pd.concat([self._data_frame, dummies], axis = 'columns')
                        
                # otherwise raising index error
                else:
                    raise IndexError(f"Error! {category_column} named column doesn't exist.")
            
        # showing this error message into screen
        except IndexError as i:
            print(f"\nError message from FeatureEngineering class's one_hot_encoder() method: {i}")
            
        # showing this error message into screen
        except Exception as e:
            print(f"\nError message from FeatureEngineering class's one_hot_encoder() method: {e}")



"""### Class DataSplit inheriting FeatureEngineering class:

purpose:
        1. To extract the features and and label from dataframe.
        2. To split the data into 2 category, one for training set
         another one for testing set.
        
    ** Approach: We will follow 80-20 approach, where 80% data will
              be for model training and 20% will be for model testing."""
class DataSplit(FeatureEngineering):
    # protected class attributes for training feature and label
    _training_features : pd.DataFrame = None 
    _training_label : pd.DataFrame = None
    
    # protected class attributes for training feature and label
    _testing_features : pd.DataFrame = None
    _testing_label : pd.DataFrame = None
        
    
    """** Method feature_and_label_conversion() returns None:
    
    purpose: 1. Extracting the features from the dataframe and converting into dataframe.
             2. Extracting the label from the dataframe and converting into dataframe."""
    def feature_and_label_extraction(self) -> None:
        # for handling the error here
        try:      
            # extracting the label from dataframe and converting into dataframe
            self._label = pd.DataFrame(self._data_frame.loc[:, self._label])
            
            # dropping the label column from the dataframe
            self._data_frame.drop(self._label, axis = 'columns', inplace = True)
            
            # extracting the features from the dataframe
            self._features = self._data_frame
                
            
            """calling the data_frame_splitting() method
            for splitting the dataframe into two parts"""
            self.data_frame_splitting()
            
        # showing this error message into screen
        except Exception as e:
            print(f"\nError message from DataSplit class's feature_and_label_extraction() method: {e}")
    
    
    """** Method data_frame_splitting() returns None:
    
    purpose: For splitting the dataframe into 80-20 split"""
    def data_frame_splitting(self) -> None:
        # for handling the error here
        try:
            """splitting the features and labels with the given split_range for training data"""
            self._training_features = self._features.loc[:self._split_range , :]
            self._training_label = self._label.loc[:self._split_range , :]
            
            """splitting the features and labels with the given split_range for testing data"""
            self._testing_features = self._features.loc[self._split_range: , ]
            self._testing_label = self._label.loc[self._split_range: , :]
            
            # showing for acknowledgement
            print("\nDataset has splitted succesfully.")
            
        # showing this error message into screen
        except Exception as e:
            print(f"\nError message from DataSplit class's data_frame_splitting() method: {e}")


    
"""### Class TrainModelAndPrediction inheriting DataSplit class.

purpose: 
        1. To train the model with 80% splitted data.
        2. To predict the model using those 20% splitted data."""
class TrainModelAndPrediction(DataSplit):
    # creating private instance of LinearRegression() class
    __multi_linear_regression = linear_model.LinearRegression()
    # to store the predicted output here
    predicted_output : np.array = None
    
    """**Method train_model() returns None.
    
    purpose: To train the model with features and label"""
    def train_model(self) -> None:        
        # for handling the error here
        try:
            # fitting the model with features and label 
            self.__multi_linear_regression.fit(self._training_features, self._training_label)    
            
            # showing for acknowledgement
            print(f"\nModel has trained successfully.\n\nModel's training score: {self.__multi_linear_regression.score(self._training_features, self._training_label) * 100:.2f} %")
            
        # showing this error message into screen
        except Exception as e:
            print(f"\nError message from TrainModelAndPrediction class's train_model() method: {e}")
            
    
    """**Method predict_value() returns None.
    
    purpose: To predict the value from trained model by giving
             features."""
    def predict_value(self, features_args : dict) -> None:
        # for handling the error here
        try:
            """Testing perform start"""
            # giving the features to model to predict the label
            self.predicted_output = self.__multi_linear_regression.predict(self._testing_features)
            # concating the original dataframe with new price column wise            
            self._original_df["Predicted Price"] = self.predicted_output
            # saving into csv file for checking the price
            self._original_df.to_csv("testing_model.csv", index = False)
            """Testing perform end"""
            
            """original prediction start"""
            # checking type first
            if (type(features_args) == dict):
                # sending to preprocess function as features
                features : pd.DataFrame = self.preprocess_new_data(pd.DataFrame(features_args))
                # creating a features copy to assign into file output
                _save_features : pd.DataFrame = pd.DataFrame(features_args)
                # Ensure new data has the same columns as the training data
                features = features.reindex(columns = self._training_features.columns, fill_value = 0)
                
                # making the predictions
                prediction = self.__multi_linear_regression.predict(features)
                
                # assigning predicted price into features dataframe
                _save_features["Predicted Price"] = prediction
                
                # assigning predicted price csv file into folder
                _save_features.to_csv("Predicted_output.csv", index = False)
                
                # showing for acknowledgement
                print("\nPrice predicted successfully!\n-> Check the folder with Predicted_output.csv named file")
            
            # otherwise showing type error    
            else:
                raise TypeError("Error! only dictionary is accepted.")
            """original prediction end"""

        # showing this error message into screen
        except Exception as e:
            print(f"\nError message from TrainModelAndPrediction class's predict_value() method: {e}")
            
    
    """** Method preprocess_new_data that returns None.
    
    purpose: To take the user's new data as argument and
             pre-process again to give for prediction to model."""
    def preprocess_new_data(self, new_data : pd.DataFrame) -> None:
        # for handling the error here
        try:
            """For one hot encoding the categorical columns"""
            # first checking if the column exist or not in the features using loop
            for category_column in self._one_hot_encoding_column:
                # checking in features that exist or not
                if category_column in new_data.columns:
                    """creating dummies of that columns by removing the first column 
                    because off multilinear trap"""
                    dummies = pd.get_dummies(new_data[category_column], 
                                             dtype = int, drop_first = True)
                    
                    # droping that category column from the features
                    new_data.drop([category_column], axis = 1, inplace = True)
                    
                    # concating the dummy column and updating orinal features column
                    new_data = pd.concat([new_data, dummies], axis = 'columns')
                        
                # otherwise raising index error
                else:
                    raise IndexError(f"Error! {category_column} named column doesn't exist.")
            
            
            """For label encoding the column"""    
            # first checking if the column exist or not in the features
            if self._label_encoding_column in new_data.columns:
                # label encoding the that column by using fit_transform() method from LabelEncoder class
                new_data[self._label_encoding_column] = self._label_encoder.fit_transform(
                                                            new_data[self._label_encoding_column])
                
            # otherwise raising index error
            else:
                raise IndexError(f"Error! {self._label_encoding_column} named column doesn't exist.")
            
            # returing the pre-processed data
            return new_data
            
            
        # showing this error message into screen
        except IndexError as i:
            print(f"\nError message from TrainModelAndPrediction class's preprocess_new_data() method: {i}")
            
        # showing this error message into screen
        except Exception as e:
            print(f"\nError message from TrainModelAndPrediction class's preprocess_new_data() method: {e}")



"""### Class PlotGraph inherits TrainModelAndPrediction.

Purpose: To draw a graph comparing previous price and 
         predicted price."""
class PlotGraph(TrainModelAndPrediction):
    def plot_graph(self):
        # for handling the error here
        try:
            # actual price
            actual = self._testing_label.values.ravel()[0:5]   # Convert to 1D array
            # predicted price
            predicted = self.predicted_output.ravel()[0:5]   # convert to 1D array 
            
            # for the indices for x axis
            indices = np.arange(5)
            
            # for using custom style
            plt.style.use("seaborn-v0_8-white")

            # Plot grouped bar chart 
            # for actual price
            plt.bar(indices - 0.2, actual, width=0.4, 
                    label="Actual Price", color='lightgreen')
            # for predicted price
            plt.bar(indices + 0.2, predicted, width=0.4, 
                    label="Predicted Price", color='seagreen')
            
            plt.legend() # for showing the labels
            plt.xlabel("Sample Index", fontsize = 18, fontweight = "bold") # for xlabel
            plt.ylabel("Car Price", fontsize = 18, fontweight = "bold") # for ylabel
            plt.title("Actual vs. Predicted Prices", fontsize = 20, fontweight = "bold") # for title
            plt.show() # for showing the graph
             
        
        # showing this error message into screen
        except Exception as e:
            print(f"\nError message from PlotGraph class's plot_graph() method: {e}")        



"""Checking the file is main file or not.

purpose: To check before executing the program, that
        the file is main file or the classes are using 
        in any other file."""
if __name__ == "__main__":
    # for handling the error here
    try:
        # new features dictionary
        features = {
                    "Brand": ["Toyota", "Honda", "BMW", "Ford", "Hyundai"],
                    "Model": ["Corolla", "Civic", "X5", "Mustang", "Elantra"],
                    "Year": [2019, 2020, 2018, 2021, 2017],
                    "Engine_Size": [1.8, 2.0, 3.0, 5.0, 1.6],
                    "Fuel_Type": ["Petrol", "Diesel", "Petrol", "Petrol", "Diesel"],
                    "Transmission": ["Automatic", "Manual", "Automatic", "Manual", "Automatic"],
                    "Mileage": [25000, 30000, 40000, 15000, 50000],
                    "Doors": [4, 4, 5, 2, 4],
                    "Owner_Count": [1, 2, 1, 3, 2]
                    }
        
        # creating an instance and passing the required values
        obj = PlotGraph(dataset= "car_price_dataset.csv",
                        features= ["Brand", "Model", "Year", 
                                "Engine_Size", "Fuel_Type", 
                                "Transmission", "Mileage", 
                                "Doors", "Owner_Count"],
                        label = "Price",
                        split_range = 8000,
                        label_encoding_column= "Model",
                        one_hot_encoding_column=["Brand", "Fuel_Type", "Transmission"])
        
        # calling this method for cleaning the data
        obj.data_cleaning()
        
        # calling feature_engineer() method for applying feature engineering
        obj.feature_engineer()
        
        # calling this method for splitting the dataframe
        obj.feature_and_label_extraction()
        
        # calling this method for training the model first
        obj.train_model()
        
        # calling the predict_value() method to predict the output
        obj.predict_value(features_args = features)
        
        # for plotting the graph to screen
        obj.plot_graph()
    
    # showing this error message into screen
    except Exception as e:
        print(f"\nError message from TrainModelAndPrediction class's predict_value() method: {e}")
    