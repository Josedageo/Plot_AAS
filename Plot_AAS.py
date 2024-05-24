import streamlit as st
import pandas as pd
import plotly.express as px

# Define the correct file paths using relative paths
hydrographs_path = './compiled_HydroGraphs.csv'
kvalues_path = './Compiled_Kvalues.csv'
control_points_path = './control_points.csv'

# Load the data
hydrographs_df = pd.read_csv(hydrographs_path)
kvalues_df = pd.read_csv(kvalues_path)
control_points_df = pd.read_csv(control_points_path)

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

# Adjust the merge logic based on actual column names
# Assuming the column names might be slightly different, adjust accordingly
merge_columns = list(set(hydrographs_df.columns) & set(kvalues_df.columns))
merged_df = pd.merge(hydrographs_df, kvalues_df, on=merge_columns)
merged_df = pd.merge(merged_df, control_points_df, left_on='Label_unique', right_on='ID')

# Debug: Display merged data info
if st.sidebar.checkbox("Show Merged Data Info"):
    st.write("Merged Dataframe")
    st.write(merged_df.head())
    st.write("Columns: ", merged_df.columns.tolist())

# Streamlit app
st.title('Plot_AAS')

# Sidebar for selecting plot parameters
plot_type = st.sidebar.selectbox('Select Plot Type', ['scatter', 'line', 'bar', 'area', 'pie', 'histogram', 'box', 'violin', 'surface', 'heatmap'])
x_axis = st.sidebar.selectbox('Select X Axis', merged_df.columns)
y_axis = st.sidebar.selectbox('Select Y Axis', merged_df.columns)
z_axis = st.sidebar.selectbox('Select Z Axis (if applicable)', [None] + list(merged_df.columns))

# Filtering options
filter_columns = st.sidebar.multiselect('Filter Columns', merged_df.columns)
filters = {}
for column in filter_columns:
    unique_values = merged_df[column].unique()
    selected_values = st.sidebar.multiselect(f'Select values for {column}', unique_values, default=unique_values)
    filters[column] = selected_values

# Apply filters
filtered_df = merged_df.copy()
for column, values in filters.items():
    filtered_df = filtered_df[filtered_df[column].isin(values)]

# Debug: Display filtered data info
if st.sidebar.checkbox("Show Filtered Data Info"):
    st.write("Filtered Dataframe")
    st.write(filtered_df.head())
    st.write("Number of rows after filtering: ", len(filtered_df))

# Aggregation options
aggregation_function = st.sidebar.selectbox('Select Aggregation Function', ['None', 'Max', 'Min', 'Average', 'Sum'])
if aggregation_function != 'None':
    agg_df = filtered_df.groupby([x_axis]).agg({y_axis: aggregation_function.lower()})
    filtered_df = filtered_df.merge(agg_df, on=[x_axis], suffixes=('', f'_{aggregation_function.lower()}'))
    y_axis += f'_{aggregation_function.lower()}'

# Debug: Display aggregated data info
if st.sidebar.checkbox("Show Aggregated Data Info"):
    st.write("Aggregated Dataframe")
    st.write(filtered_df.head())
    st.write("Number of rows after aggregation: ", len(filtered_df))

# Aesthetics options
st.sidebar.header('Graph Aesthetics')
title = st.sidebar.text_input('Title', 'Graph Title')
x_axis_label = st.sidebar.text_input('X Axis Label', x_axis)
y_axis_label = st.sidebar.text_input('Y Axis Label', y_axis)
show_legend = st.sidebar.checkbox('Show Legend', True)
show_grid = st.sidebar.checkbox('Show Grid', True)
marker_size = st.sidebar.slider('Marker Size', 1, 20, 10)
line_width = st.sidebar.slider('Line Width', 1, 10, 2)
color = st.sidebar.color_picker('Color', '#00f900')

# Plotting
fig = None
if plot_type == 'scatter':
    fig = px.scatter(filtered_df, x=x_axis, y=y_axis)
    fig.update_traces(marker=dict(size=marker_size, color=color))
elif plot_type == 'line':
    fig = px.line(filtered_df, x=x_axis, y=y_axis)
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
    fig = px.histogram(filtered_df, x=x_axis, y=y_axis)
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
else:
    st.write('Select a valid plot type and axes.')
