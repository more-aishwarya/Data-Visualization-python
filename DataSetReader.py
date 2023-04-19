import pandas as pd
import seaborn as sb
import matplotlib.pyplot as mp
import numpy as np
from IPython.display import Image, HTML, display
from PIL import Image
from io import BytesIO
import requests


class DataSetReader:
#  handle exceptions in the functions.  
    def handle_exceptions(func): 
        def wrapper(*args, **kwargs): 
            '''Function to handle exeptions'''
            try: 
                return func(*args, **kwargs) 
            except Exception as e: 
                print(f"An exception occurred: {e}") 
        return wrapper 

#Loading data in to DataFrames for CSV and JSON.
    @handle_exceptions
    def read_data(self):
        '''Read data from csv'''
        df = pd.read_csv('udacity.csv')
        self.Data = df 
        return self.Data
    
    @handle_exceptions
    def read_data_JSON(self):
        '''Read data from JSON File'''
        df1 = pd.read_json('udacity.json')
        self.jsonData = df1 
        return self.jsonData

# Check the Data Types of your data columns. 
    @handle_exceptions  
    def checkDataType(self):
        '''Function to check the data types of the colums'''
        df = self.Data.info()
        print(df)

#Drop any NULL values .
    @handle_exceptions   
    def drop_data(self): 
        '''Drop Null values '''
        df=self.Data
        droped_value= df.dropna(subset= ['Type','Rating']) 
        return droped_value 
    
#Check for missing values
    @handle_exceptions
    def missing_values(self): 
        '''Display Missing value count for columns in tabular form'''
        df = self.Data
        mis_val = df.isnull().sum() 
        mis_val_percent = 100 * df.isnull().sum() / len(df) 
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1) 
        mis_val_table_ren_columns = mis_val_table.rename( 
        columns = {0 : 'Missing Values', 1 : '% of Total Values'}) 
        mis_val_table_ren_columns = mis_val_table_ren_columns[ 
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values( 
        '% of Total Values', ascending=False).round(1) 
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"       
            "There are " + str(mis_val_table_ren_columns.shape[0]) + 
              " columns that have missing values.") 
        return mis_val_table_ren_columns 
      
# Drop duplicate values.   
    @handle_exceptions
    def drop_duplicates(self): 
        '''Drop duplicate values from a column in dataset'''
        df=self.Data
        dropped_duplicate_value = df.drop_duplicates(subset='Prerequisites') 
        return dropped_duplicate_value
    
# Check for outliers using a histogram.    
    @handle_exceptions
    def getHistogram(self):
        '''Get Histogram for Rating Column'''
        df = self.Data
        data = df['Rating']
        mp.hist(data, color='orange', edgecolor='black')
        mp.title('Histogram of Rating')
        mp.show()

#Check for outliers using a box plot. 
    @handle_exceptions
    def getBoxPlot(self):
        '''Get Box plot for review and rating count column values '''
        df = self.Data
        df1 = df[['Rating', 'Review Count']] 
        sb.boxplot(data=df1, width=0.5,fliersize=5) 

# Plot features against each other using a pair plot.
    @handle_exceptions
    def getPairPlot(self):
        '''Get pair plot'''
        df = self.Data
        sb.pairplot(df)

#Use a HeatMap for finding the correlation between the features(Feature to Feature)    
    @handle_exceptions
    def getHeatMap(self):
        '''Generate heatmap for finding correlation between features'''
        df = self.Data
        corr_matrix = df.corr()
        sb.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        mp.title('Correlation Heatmap')
        mp.show()

#Use a scatter plot to show the relationship between 2 variables. 
    @handle_exceptions
    def getScatterPlot(self):
            '''Get scatter plot to show relationship between 2 variables'''
            df = self.Data
            mp.scatter(df['Review Count'],
            df['Rating'],marker='D',color='red',label='Very Low')
            mp.scatter(df['Rating'],
            df['Rating'],marker='o',color='blue',label='Low')
            mp.title('Scatter plot for Review count and rating')
            mp.xlabel('Review count')
            mp.ylabel('Rating')
            mp.legend()


#Merging two Data Frames 
    @handle_exceptions
    def mergeDataframes(self):
           '''Merge 2 data frames'''
           df=self.Data
           otherdf=pd.read_csv('LinkedlnLearning.csv')
           merged_df = pd.merge(df, otherdf, on='Title', how='inner')
           merged_df.to_csv('mergeddata', index=False)
           return merged_df

# Perform query on data 
    @handle_exceptions
    def getDataBasedOnQuery(self):
         '''Get data based on query for a column '''
         df = self.Data
         DurationFilter=df.query('Duration == "4 Weeks"')
         print(DurationFilter)


#Representing data in matrix form.
    @handle_exceptions  
    def getMatrix(self):
         '''represent data in  matrix form '''
         df = self.Data
         df_matrix =np.asmatrix(df)
         print(df_matrix)  

#Upload data to Numerical Python (NumPy)  
    @handle_exceptions
    def uploadNumericalDataToPython(self):
         '''Upload data to numerical python'''
         df = self.Data
         np_df = df.to_numpy()
         np.save('numpysave', np_df)
         return np_df

# Select a slice or part of the data and display. 
    @handle_exceptions
    def sliceData_iloc(self):
         '''Slice data or part of data'''
         df = self.Data
         df_slice=df.iloc[0:180]
         return df_slice
    
    @handle_exceptions
    def slice_data_loc(self): 
         '''Slice data or part of data'''
         df = self.Data
         df_slice_courses = df.loc[(df["Level"] == "advanced"),['Title','Rating']]
         return df_slice_courses
    
#Use conditions and segregate the data based on the condition (like show data of a feature(column) >,<,= a number)      
    @handle_exceptions
    def getDataBasedOnCondition(self):
         '''Get data based on conditions'''
         df = self.Data
         ratings=df[df['Rating'] >= 4.8]
         print("Courses with ratings greater than 4.8 are :")
         return ratings

#Use mathematical and statistical functions using libraries.  
    # Calculate the mean
    @handle_exceptions
    def calculateMean(self):
       '''Calculate mean for review count column'''
       df = self.Data
       mean = np.mean(df['Review Count'])
       print(mean)

    @handle_exceptions
    def calculateMedian(self):
       '''Calculate median for Review count column'''
       df = self.Data
       median = np.median(df['Review Count'].dropna())
       print  (median)

    # Calculate the standard deviation
    @handle_exceptions
    def calculateStandardDeviation(self):
       '''Calculate median for Review count column'''
       df = self.Data
       std_dev = np.std(df['Review Count'])
       print(std_dev)
   
 #Other mathematical Functions 
    # Count the number of rows in the DataFrame 
    @handle_exceptions
    def countRowsInDataFrame(self):
        '''Calculate number of rows in dataframe'''
        df = self.Data
        num_rows = len(df) 
        print('Number of rows in the DataFrame: ',num_rows) 

    # Calculate the Sum of values of a column in dataset
    @handle_exceptions
    def calculateSumOfReviewCount(self):
         '''Calculate sum of values of  Review count column'''
         df = self.Data
         reviewtotal=df.sum(numeric_only=True)
         print(reviewtotal['Review Count']) 
    # Calculate the Min and Max value for a column in dataset
    @handle_exceptions
    def getMin(self):
         '''Get Min value for  Review count column'''
         df = self.Data
         print('Minimum Review Count')
         min_review=df.min(numeric_only=True)
         print(min_review['Review Count']) 
   
    @handle_exceptions
    def getMax(self):
         '''Get Max value for  Review count column'''
         df = self.Data
         print('Maximum Rating Count')
         max_review=df.max(numeric_only=True)
         print(max_review['Review Count']) 

#Search & display particular course based on title
    @handle_exceptions
    def searchCourseByTitle(self):
        '''Search course by tittle'''
        df = self.Data
        searched_course=(df.loc[df['Title'] == 'Digital Marketing'])
        return searched_course
    

#Display images passed as URL in dataset instructor column
    @handle_exceptions
    def getImageFromDataset(self):
        '''Function to get image from csv passed as URL in Instructor column'''
        df=self.Data
        for col in df.columns:
              if df[col].dtype == 'object':
                if df[col].str.startswith('http').any():
                 for index, value in df[col].items():
                    if value.startswith('http'):
                      response = requests.get(value)
                      image_data = response.content
                      image = Image.open(BytesIO(image_data))
                      display(image)
                      print(f"{df.iloc[index].drop(col)}")
                      print("="*60)

#Use of args and kwargs on datset 
    
    @handle_exceptions
    def argsfunc(self, *args): 
        '''Use of args '''
        df = self.Data 
        matching_rows = [] 
        for _, row in df.iterrows(): 
            if any(str(arg) in row.astype(str).values for arg in args): 
                matching_rows.append(row) 
        return matching_rows 

    @handle_exceptions
    def kwargsfunc(self, **kwargs): 
        '''Use of kwargs'''
        df = self.Data 
        for _, row in df.iterrows(): 
            for key, value in kwargs.items(): 
                if str(value) in row.astype(str).values: 
                    return row.to_dict() 
                
#Top 3 courses based on ratings 
    @handle_exceptions
    def topCourses(self):
         '''Display top courses'''
         df = self.Data
         courses=df[df['Rating'] >= 4.8]
         print("Top 3 courses are  :")
         return courses
        