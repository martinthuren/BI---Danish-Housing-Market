import pandas as pd
import numpy as np
import re
import seaborn as sns
from sklearn.linear_model import LinearRegression
from prophet import Prophet
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import streamlit as st
from kneed import KneeLocator



# Add introductory text and image
st.title('Welcome to The Housing Prices App')
st.write('This app allows you to visualize and analyze housing prices data.')

# Add an image if desired
# st.image('your_image.jpg', caption='Housing Prices', use_column_width=True)

# Add a section for the app's description
st.header('App Description')
st.write("""
This app provides various visualizations and analyses of housing prices data. You can choose from the following visualization types:
- Average Housing Prices Comparison
- Linear Regression Model
- Heatmap of Housing Prices Over Time
- Time Series Forecasting
""")

# Add a section for instructions on how to use the app
st.header('How to Use')
st.write("""
1. Select a visualization type from the dropdown menu.
2. Follow the instructions provided for each visualization type to interact with the data.
3. Explore different features and options to gain insights into housing prices trends.
""")

# Add a section for additional information or credits
st.header('Additional Information')
st.write("""
- Data Source: Finans Danmark
- Developed by: Philip and Martin
- GitHub Repository: (https://github.com/martinthuren/BI---Danish-Housing-Market)
""")


# Add a separator to distinguish the front page from the main content
st.markdown('---')

# Now you can include the rest of your code for visualizations and analyses
# Paste your existing code here



# Disable the warning about the use of st.pyplot() without passing any arguments
st.set_option('deprecation.showPyplotGlobalUse', False)

def clean_string(input_string):
    # Remove non-alphanumeric characters and whitespace
    cleaned_string = re.sub(r'[^a-zA-Z0-9\s]', '', input_string)
    # Remove leading and trailing whitespace
    cleaned_string = cleaned_string.strip()
    return cleaned_string

# Reading data
ejer_file_path = "/Users/martinthuren/Desktop/ejerlejlighed.xlsx"
parcelrække_file_path = "/Users/martinthuren/Desktop/parcelogrækkehuse.xlsx"

try:
    ejer = pd.read_excel(ejer_file_path, index_col=0)
    parcelogrække = pd.read_excel(parcelrække_file_path, index_col=0)
except Exception as e:
    st.error(f"An error occurred while reading the data: {e}")
    st.stop()

# Replace ".." with NaN
ejer.replace("..", np.nan, inplace=True)
parcelogrække.replace("..", np.nan, inplace=True)

# Replace 0 with NaN
ejer.replace(0, np.nan, inplace=True)
parcelogrække.replace(0, np.nan, inplace=True)

# Convert NaN values to 0
ejer.fillna(0, inplace=True)
parcelogrække.fillna(0, inplace=True)

# Clean up city names and zip codes
ejer.index = ejer.index.map(clean_string)
parcelogrække.index = parcelogrække.index.map(clean_string)

# Add interactivity
st.title('The Housing Prices App')

# Select visualization type
visualization_type = st.selectbox(
    'Select Visualization Type:',
    ['Average Housing Prices Comparison', 'Linear Regression Model', 'Heatmap of Housing Prices Over Time', 'Time Series Forecasting','K Means Clustering']
)

if visualization_type == 'Average Housing Prices Comparison':
    # Extract years from column names
    column_years_ejer = ejer.columns.astype(str).str.extract(r'(\d+)K\d+').astype(float).squeeze()
    valid_years_ejer = column_years_ejer.dropna().astype(int)

    # Check if there are valid years
    if not valid_years_ejer.empty:
        # Selecting quarter using slider
        selected_kvartal = st.slider('Select Quarter:', min_value=1, max_value=4, value=1)

        # Initialize lists to store average prices and years
        average_prices_ejer = []
        average_prices_parcelogrække = []
        years = []

        # Iterate over valid years to calculate average prices
        for year in valid_years_ejer:
            selected_column = f"{year}K{selected_kvartal}"
            if selected_column in ejer.columns and selected_column in parcelogrække.columns:
                average_ejer_prices = ejer[selected_column].mean()
                average_parcelogrække_prices = parcelogrække[selected_column].mean()

                average_prices_ejer.append(average_ejer_prices)
                average_prices_parcelogrække.append(average_parcelogrække_prices)
                years.append(year)

        # Create a DataFrame for prices
        prices_df = pd.DataFrame({
            'Year': years,
            'Parcel- og rækkehus': average_prices_ejer,
            'Ejerlejlighed': average_prices_parcelogrække
        })

        # Melt the DataFrame to have 'Year', 'Type', and 'Average Price' columns
        prices_df = prices_df.melt(id_vars='Year', var_name='Type', value_name='Average Price')

        # Plot the bar chart
        fig, ax = plt.subplots(figsize=(17, 8))
        ax = sns.barplot(x='Year', y='Average Price', hue='Type', data=prices_df)

        # Add annotations for hover-over price display
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=10, color='black', xytext=(0, 5), textcoords='offset points')

        # Set plot title and axis labels
        plt.title('Average Housing Prices Comparison Per Square Meter')
        plt.xlabel('Year')
        plt.ylabel('Average Housing Prices')

        # Show the plot
        st.pyplot(fig)
    else:
        st.error("No valid years found.")




elif visualization_type == 'Linear Regression Model':
    # Filter out NaN values before computing min and max
    column_years_ejer = ejer.columns.astype(str).str.extract(r'(\d+)K\d+').astype(float).squeeze()
    valid_years_ejer = column_years_ejer.dropna()
    
    if not valid_years_ejer.empty:
        selected_year = st.slider('Select Year:', min_value=int(valid_years_ejer.min()), max_value=int(valid_years_ejer.max()))
        
        # Tilføj en slider for at vælge kvartal
        selected_kvartal = st.slider('Vælg Kvartal:', min_value=1, max_value=4, value=1)

        # Generer kolonnenavn baseret på brugerens valg af år og kvartal
        selected_column = f"{selected_year}K{selected_kvartal}"
        
        if selected_column in ejer.columns and selected_column in parcelogrække.columns:
            X_ejer = ejer[selected_column].values.reshape(-1, 1)
            y_ejer = parcelogrække[selected_column].values
            
            X_parcelogrække = parcelogrække[selected_column].values.reshape(-1, 1)
            y_parcelogrække = ejer[selected_column].values

            # Fit linear regression model
            model_ejer = LinearRegression()
            model_ejer.fit(X_ejer, y_ejer)
            
            model_parcelogrække = LinearRegression()
            model_parcelogrække.fit(X_parcelogrække, y_parcelogrække)

            # Plot data and regression line
            plt.figure(figsize=(10, 6))
            plt.scatter(X_ejer, y_ejer, color='skyblue', label='Ejerlejligheder')
            plt.scatter(X_parcelogrække, y_parcelogrække, color='orange', label='Parcel- og rækkehuse')
            plt.plot(X_ejer, model_ejer.predict(X_ejer), color='blue', label='Ejerlejligheder Regression Line')
            plt.plot(X_parcelogrække, model_parcelogrække.predict(X_parcelogrække), color='red', label='Parcel- og rækkehuse Regression Line')
            plt.title(f'Average Housing Prices Comparison Per Square Meter ({selected_column})')
            plt.xlabel('Ejerlejlighed Prices')
            plt.ylabel('Parcel- og rækkehus Prices')
            plt.legend()
            st.pyplot()
        else:
            st.error("Valgte kolonne findes ikke i DataFrames.")
    else:
        st.error("No valid years found.")

elif visualization_type == 'Heatmap of Housing Prices Over Time':
    # Highlight specific data points
    highlight_data = st.checkbox('Highlight Specific Data Points')

    # Transpose the DataFrame for plotting
    ejer_transposed = ejer.T

    # Heatmap showing housing prices for each zip code over time
    plt.figure(figsize=(10, 6))
    if highlight_data:
        # Add custom highlighting logic here
        pass  # Placeholder for custom highlighting logic
    sns.heatmap(ejer_transposed, cmap='YlGnBu', cbar_kws={'label': 'Housing Prices'})
    plt.title('Heatmap of Housing Prices Over Time by Zip Code')
    plt.xlabel('Zip Code')
    plt.ylabel('Quarter')
    st.pyplot()

elif visualization_type == 'Time Series Forecasting':
    # Step 1: Read the dataset from an Excel file
    excel_file = '/Users/martinthuren/Desktop/ejerlejlighed.xlsx'
    df = pd.read_excel(excel_file, index_col=0)

    # Step 2: Prepare the data for time series forecasting
    # Transpose the DataFrame to have time series as rows and features as columns
    df = df.transpose()

    # Convert the index to datetime type using a custom parser
    def custom_date_parser(date_string):
        year = date_string[:4]
        quarter = date_string[-2:]
        if quarter == 'K1':
            month = 3
        elif quarter == 'K2':
            month = 6
        elif quarter == 'K3':
            month = 9
        else:
            month = 12
        return pd.to_datetime(f'{year}-{month}-01')
    

    df.index = df.index.map(custom_date_parser)

    # Step 3: Train a forecasting model with Prophet
    # Rename columns as required by Prophet
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'ds', '1000-1499 Kbh.K.': 'y'}, inplace=True)

    # Initialize Prophet model
    model = Prophet()

    # Fit the model to the data
    model.fit(df)

    # Step 4: Forecast future values
    # Function to convert months slider value to periods
    def months_to_periods(months):
        return months * 30

    # Slider for selecting months
    selected_months = st.slider('Select Number of Months for Forecasting:', min_value=1, max_value=24, value=12)

    # Generate future time indices for forecasting
    future_indices = pd.date_range(start=df['ds'].iloc[-1], periods=months_to_periods(selected_months), freq='D')

    # Create a DataFrame with future indices
    future_df = pd.DataFrame({'ds': future_indices})

    # Make predictions for future values
    forecast = model.predict(future_df)

    # Step 5: Visualize the forecasted values
    st.write("### Forecasted Values:")
    st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

    plt.figure(figsize=(10, 6))
    model.plot(forecast, xlabel='Date', ylabel='1000-1499 Kbh.K.', ax=plt.gca())
    plt.title('Time Series Forecasting with Prophet')
    plt.grid(True)
    st.pyplot()

elif visualization_type == 'K Means Clustering':
    st.header('K-Means Clustering Visualization')

    # Read the data from Excel file
    excel_file = '/Users/martinthuren/Desktop/ejerlejlighed.xlsx'
    df = pd.read_excel(excel_file)

    # Set the first row as the column names
    df.columns = df.iloc[0]

    # Drop the first row (as it's now redundant)
    df = df.drop(0)

    # Reset the index
    df.reset_index(drop=True, inplace=True)

    # Drop the 'Unnamed: 0' column if it exists
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)

    # Replace '..' with NaN
    df.replace('..', pd.NA, inplace=True)

    # Drop columns with all NaN values
    df.dropna(axis=1, how='all', inplace=True)

    # Convert columns to numeric (excluding the first column)
    df.iloc[:, 1:] = df.iloc[:, 1:].apply(pd.to_numeric)

    # Impute missing values with column means
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(df.iloc[:, 1:])

    # Apply feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Initialize a list to store inertia values for different k
    inertia_values = []

    # Try different values of k and calculate inertia
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_pca)
        inertia_values.append(kmeans.inertia_)

    # Find the optimal number of clusters using the "knee" in the elbow method curve
    kneedle = KneeLocator(range(1, 11), inertia_values, curve='convex', direction='decreasing')
    optimal_k = kneedle.elbow

    # Re-fit the KMeans model with the optimal k
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    kmeans.fit(X_pca)

    # Calculate the silhouette score
    silhouette_avg = silhouette_score(X_pca, kmeans.labels_)

    #   Plot the original clustering
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans.labels_, cmap='viridis', s=50, alpha=0.7)
    plt.title('K-Means Clustering (2D PCA)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar(label='Cluster')

    # Plot the elbow method curve
    plt.subplot(1, 2, 2)
    plt.plot(range(1, 11), inertia_values, marker='o', linestyle='--')
    plt.scatter(optimal_k, inertia_values[optimal_k-1], color='red', label=f'Optimal k ({optimal_k})')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.xticks(range(1, 11))
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    # Display the plots using st.pyplot()
    st.pyplot()

    # Display plots using Streamlit
    st.header('K-Means Clustering Visualization')

    st.write(f"The most precise number of clusters based on the elbow method: {optimal_k}")
    st.write(f"Silhouette Score: {silhouette_avg}")
