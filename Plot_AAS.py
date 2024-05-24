import streamlit as st
import pandas as pd
import plotly.express as px
from io import BytesIO
import base64
from datetime import datetime

# Ensure that the kaleido library is installed
try:
    import kaleido
except ImportError:
    st.error("Kaleido is required for saving plots as PNG. Please install it using 'pip install kaleido'.")

# Define the correct file paths using relative paths
hydrographs_path = './compiled_HydroGraphs.csv'
kvalues_path = './Compiled_Kvalues.csv'
control_points_path = './control_points.csv'

# Load the data
hydrographs_df = pd.read_csv(hydrographs_path)
kvalues_df = pd.read_csv(kvalues_path)
control_points_df = pd.read_csv(control_points_path)

# Preprocess kvalues_df to extract level and zone
kvalues_df['Level'] = kvalues_df['Label_unique'].str[:2]
kvalues_df['Zone'] = kvalues_df['Label_unique'].str[-2:]

# Reset button
if st.sidebar.button('Reset All'):
    st.experimental_rerun()

# Streamlit app
st.title('Plot_AAS')

# Display data info
if st.sidebar.checkbox("Show Data Info"):
    st.write("Hydrographs Dataframe")
    st.write(hydrographs_df.head())
    st.write("Columns: ", hydrographs_df.columns.tolist())

    st.write("Kvalues Dataframe")
    st.write(kvalues_df.head())
    st.write("Columns: ", kvalues_df.columns.tolist())

    st.write("Control Points Dataframe")
    st.write(control_points_df.head())
    st.write("Columns: ", control_points_df.columns.tolist())

# Sidebar for selecting plot parameters
file_selection = st.sidebar.selectbox('Select Data File', ['Hydrographs', 'Kvalues', 'Combined'])
plot_type = st.sidebar.selectbox('Select Plot Type', [
    'scatter', 'line', 'bar', 'area', 'pie', 'histogram', 'box', 
    'violin', 'surface', 'heatmap'
], index=5)

default_column = 'CP001' if 'CP001' in hydrographs_df.columns else hydrographs_df.columns[0]

if file_selection == 'Hydrographs':
    df = hydrographs_df
    x_axis = st.sidebar.selectbox('Select X Axis', df.columns, index=list(df.columns).index(default_column))
    y_axis = st.sidebar.selectbox('Select Y Axis', df.columns)
    z_axis = st.sidebar.selectbox('Select Z Axis (if applicable)', [None] + list(df.columns))
elif file_selection == 'Kvalues':
    df = kvalues_df
    x_axis = st.sidebar.selectbox('Select X Axis', df.columns)
    y_axis = st.sidebar.selectbox('Select Y Axis', df.columns)
    z_axis = st.sidebar.selectbox('Select Z Axis (if applicable)', [None] + list(df.columns))
else:
    # Check if 'Label_unique' exists in both DataFrames
    if 'Label_unique' in hydrographs_df.columns and 'Label_unique' in kvalues_df.columns:
        df = pd.merge(hydrographs_df, kvalues_df, on='Label_unique', suffixes=('_hydro', '_kvalues'))
        x_axis = st.sidebar.selectbox('Select X Axis', df.columns)
        y_axis = st.sidebar.selectbox('Select Y Axis', df.columns)
        z_axis = st.sidebar.selectbox('Select Z Axis (if applicable)', [None] + list(df.columns))
    else:
        st.error("The column 'Label_unique' does not exist in one or both DataFrames.")

# Filtering options
with st.sidebar.expander('Filter Options', expanded=False):
    filter_columns = st.multiselect('Filter Columns', df.columns if 'df' in locals() else [])
    filters = {}
    for column in filter_columns:
        unique_values = df[column].unique()
        selected_values = st.multiselect(f'Select values for {column}', unique_values, default=unique_values)
        filters[column] = selected_values

    # Apply filters
    if 'df' in locals():
        filtered_df = df.copy()
        for column, values in filters.items():
            filtered_df = filtered_df[filtered_df[column].isin(values)]

        # Remove outliers based on the average value
        percentile_filter = st.checkbox("Remove Outliers")
        if percentile_filter:
            outlier_threshold = st.slider('Select Standard Deviation Threshold', 1, 3, 2)
            if y_axis in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df[y_axis]):
                mean_value = filtered_df[y_axis].mean()
                std_dev = filtered_df[y_axis].std()
                lower_bound = mean_value - outlier_threshold * std_dev
                upper_bound = mean_value + outlier_threshold * std_dev
                filtered_df = filtered_df[(filtered_df[y_axis] >= lower_bound) & (filtered_df[y_axis] <= upper_bound)]

        # Display filtered data info
        if st.checkbox("Show Filtered Data Info"):
            st.write("Filtered Dataframe")
            st.write(filtered_df.head())
            st.write("Number of rows after filtering: ", len(filtered_df))

        # Aggregation options
        aggregation_function = st.selectbox('Select Aggregation Function', ['None', 'Max', 'Min', 'Average', 'Sum'])
        if aggregation_function != 'None':
            agg_df = filtered_df.groupby([x_axis]).agg({y_axis: aggregation_function.lower()})
            filtered_df = filtered_df.merge(agg_df, on=[x_axis], suffixes=('', f'_{aggregation_function.lower()}'))
            y_axis += f'_{aggregation_function.lower()}'

        # Display aggregated data info
        if st.checkbox("Show Aggregated Data Info"):
            st.write("Aggregated Dataframe")
            st.write(filtered_df.head())
            st.write("Number of rows after aggregation: ", len(filtered_df))

# Lookup and merge additional data dynamically for Hydrographs file
if file_selection == 'Hydrographs' and 'Label_unique' in filtered_df.columns:
    filtered_df = filtered_df.merge(control_points_df[['ID', 'Zone', 'Level']], left_on='Label_unique', right_on='ID', how='left')
    filtered_df = filtered_df.merge(kvalues_df, on=['Level', 'Zone'], how='left')

# Aesthetics options
with st.sidebar.expander('Graph Aesthetics', expanded=False):
    title = st.text_input('Title', 'Graph Title')
    x_axis_label = st.text_input('X Axis Label', x_axis if 'x_axis' in locals() else '')
    y_axis_label = st.text_input('Y Axis Label', y_axis if 'y_axis' in locals() else '')
    show_legend = st.checkbox('Show Legend', True)
    show_grid = st.checkbox('Show Grid', True)
    marker_size = st.slider('Marker Size', 1, 20, 10)
    line_width = st.slider('Line Width', 1, 10, 2)
    color = st.color_picker('Color', '#00f900')
    if plot_type == 'scatter':
        scatter_line = st.checkbox('Connect Points with Lines')

# Plotting
fig = None

if 'filtered_df' in locals():
    # Handle different plot types
    if plot_type == 'scatter':
        fig = px.scatter(filtered_df, x=x_axis, y=y_axis)
        fig.update_traces(marker=dict(size=marker_size, color=color))
        if scatter_line:
            fig.update_traces(mode='lines+markers')
    elif plot_type == 'line':
        line_shape = st.sidebar.selectbox('Line Shape', ['linear', 'spline'])
        fig = px.line(filtered_df, x=x_axis, y=y_axis, line_shape=line_shape)
        fig.update_traces(line=dict(width=line_width, color=color))
    elif plot_type == 'bar':
        fig = px.bar(filtered_df, x=x_axis, y=y_axis)
        fig.update_traces(marker_color=color)
    elif plot_type == 'area':
        fig = px.area(filtered_df, x=x_axis, y=y_axis)
        fig.update_traces(line=dict(width=line_width, color=color))
    elif plot_type == 'pie':
        fig = px.pie(filtered_df, names=x_axis, values=y_axis)
        fig.update_traces(marker=dict(colors=[color]))
    elif plot_type == 'histogram':
        fig = px.histogram(filtered_df, x=x_axis)
        y_axis_label = 'Count'
        fig.update_traces(marker_color=color)
    elif plot_type == 'box':
        fig = px.box(filtered_df, x=x_axis, y=y_axis)
        fig.update_traces(marker_color=color)
    elif plot_type == 'violin':
        fig = px.violin(filtered_df, x=x_axis, y=y_axis)
        fig.update_traces(marker_color=color)
    elif plot_type == 'surface':
        if z_axis:
            fig = px.density_contour(filtered_df, x=x_axis, y=y_axis, z=z_axis)
    elif plot_type == 'heatmap':
        if z_axis:
            fig = px.density_heatmap(filtered_df, x=x_axis, y=y_axis, z=z_axis)

    # Update layout
    if fig:
        fig.update_layout(
            title=title,
            xaxis_title=x_axis_label,
            yaxis_title=y_axis_label,
            showlegend=show_legend,
            xaxis_showgrid=show_grid,
            yaxis_showgrid=show_grid
        )
        st.plotly_chart(fig)

        # Create a download button for the plot
        try:
            buffer = BytesIO()
            fig.write_image(buffer, format='png')
            buffer.seek(0)
            b64 = base64.b64encode(buffer.read()).decode()
            filters_str = '_'.join([f"{k}={v}" for k, v in filters.items()])
            current_date = datetime.now().strftime('%Y%m%d')
            filename = f"{current_date}_{title}_{x_axis_label}_{y_axis_label}_{filters_str}.png".replace(" ", "_").replace(":", "-")
            href = f'<a href="data:file/png;base64,{b64}" download="{filename}">Download Plot as PNG</a>'
            st.markdown(href, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error generating PNG: {e}")
    else:
        st.write('Select a valid plot type and axes.')

# Additional plotting options for Kvalues and Control Points
st.sidebar.header('Additional Plots')
plot_additional = st.sidebar.checkbox('Plot Additional Data')
if plot_additional:
    additional_x_axis = st.sidebar.selectbox('Select X Axis for Additional Plot', kvalues_df.columns)
    additional_y_axis = st.sidebar.selectbox('Select Y Axis for Additional Plot', kvalues_df.columns)
    additional_df = kvalues_df.copy()
    # Apply filters based on zone and level from control_points_df
    if 'Zone' in control_points_df.columns and 'Level' in control_points_df.columns:
        selected_zone = st.sidebar.selectbox('Select Zone', control_points_df['Zone'].unique())
        selected_level = st.sidebar.selectbox('Select Level', control_points_df['Level'].unique())
        additional_df = additional_df[(additional_df['Zone'] == selected_zone) & (additional_df['Level'] == selected_level)]

    additional_fig = px.scatter(additional_df, x=additional_x_axis, y=additional_y_axis)
    st.plotly_chart(additional_fig)
