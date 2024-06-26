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
file_selection = st.sidebar.selectbox('Select Data File', ['Hydrographs', 'Kvalues'])
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
else:
    df = kvalues_df
    x_axis = st.sidebar.selectbox('Select X Axis', df.columns)
    y_axis = st.sidebar.selectbox('Select Y Axis', df.columns)
    z_axis = st.sidebar.selectbox('Select Z Axis (if applicable)', [None] + list(df.columns))

# Filtering options
with st.sidebar.expander('Filter Options', expanded=False):
    filter_columns = st.multiselect('Filter Columns', df.columns if not df.empty else [])
    filters = {}
    for column in filter_columns:
        unique_values = df[column].unique()
        selected_values = st.multiselect(f'Select values for {column}', unique_values, default=unique_values)
        filters[column] = selected_values

    # Apply filters
    if not df.empty:
        filtered_df = df.copy()
        for column, values in filters.items():
            filtered_df = filtered_df[filtered_df[column].isin(values)]

        # Percentile-based filtering
        percentile_filter = st.checkbox("Filter by Percentile")
        if percentile_filter:
            percentile = st.slider('Select Percentile', 0, 100, 95)
            if y_axis in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df[y_axis]):
                lower_bound = filtered_df[y_axis].quantile((100 - percentile) / 200)
                upper_bound = filtered_df[y_axis].quantile(1 - (100 - percentile) / 200)
                filtered_df = filtered_df[(filtered_df[y_axis] >= lower_bound) & (filtered_df[y_axis] <= upper_bound)]
                st.write(f"Debug: Lower Bound = {lower_bound}, Upper Bound = {upper_bound}")

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

