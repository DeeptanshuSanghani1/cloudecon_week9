import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="AWS Cloud Resources Dashboard",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #FF9900;
        text-align: center;
        padding: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #232F3E;
        margin-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #FF9900;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the datasets"""
    try:
        ec2_df = pd.read_csv('aws_resources_compute.csv')
        s3_df = pd.read_csv('aws_resources_S3.csv')
        
        # Clean data
        ec2_df = ec2_df.dropna(how='all')
        s3_df = s3_df.dropna(how='all')
        
        return ec2_df, s3_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

def parse_tags(tags_str):
    """Parse tags string into dictionary"""
    if pd.isna(tags_str):
        return {}
    try:
        return dict(item.split('=') for item in tags_str.split(','))
    except:
        return {}

def main():
    # Header
    st.markdown('<p class="main-header">☁️ AWS Cloud Resources Dashboard</p>', unsafe_allow_html=True)
    st.markdown("### Data Analysis - EC2 & S3 Resources")
    st.markdown("---")
    
    # Load data
    ec2_df, s3_df = load_data()
    
    if ec2_df is None or s3_df is None:
        st.error("Failed to load datasets. Please ensure CSV files are in the same directory.")
        return
    
    # Sidebar
    st.sidebar.title("📊 Dashboard Controls")
    st.sidebar.markdown("---")
    
    # Navigation
    page = st.sidebar.radio(
        "Select Analysis View:",
        ["🏠 Overview", "🖥️ EC2 Analysis", "💾 S3 Analysis", "🔧 Optimization"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 Dataset Info")
    st.sidebar.info(f"**EC2 Instances:** {len(ec2_df)}\n\n**S3 Buckets:** {len(s3_df)}")
    
    # Page routing
    if page == "🏠 Overview":
        show_overview(ec2_df, s3_df)
    elif page == "🖥️ EC2 Analysis":
        show_ec2_analysis(ec2_df)
    elif page == "💾 S3 Analysis":
        show_s3_analysis(s3_df)
    elif page == "🔧 Optimization":
        show_optimization(ec2_df, s3_df)

def show_overview(ec2_df, s3_df):
    """Display overview dashboard"""
    st.markdown("## 🏠 Overview Dashboard")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="💰 Total EC2 Cost/Hour",
            value=f"${ec2_df['CostUSD'].sum():.2f}",
            delta=f"{len(ec2_df)} instances"
        )
    
    with col2:
        st.metric(
            label="💾 Total S3 Storage",
            value=f"{s3_df['TotalSizeGB'].sum():.0f} GB",
            delta=f"{len(s3_df)} buckets"
        )
    
    with col3:
        avg_cpu = ec2_df['CPUUtilization'].mean()
        st.metric(
            label="📊 Avg CPU Utilization",
            value=f"{avg_cpu:.1f}%",
            delta="EC2 Instances"
        )
    
    with col4:
        st.metric(
            label="💵 Total S3 Cost/Month",
            value=f"${s3_df['CostUSD'].sum():.2f}",
            delta="Monthly"
        )
    
    st.markdown("---")
    
    # Regional distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🌍 EC2 Cost by Region")
        ec2_region_cost = ec2_df.groupby('Region')['CostUSD'].sum().sort_values(ascending=False)
        fig = px.bar(
            x=ec2_region_cost.index,
            y=ec2_region_cost.values,
            labels={'x': 'Region', 'y': 'Total Cost (USD/hour)'},
            color=ec2_region_cost.values,
            color_continuous_scale='Blues'
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🌍 S3 Storage by Region")
        s3_region_storage = s3_df.groupby('Region')['TotalSizeGB'].sum().sort_values(ascending=False)
        fig = px.bar(
            x=s3_region_storage.index,
            y=s3_region_storage.values,
            labels={'x': 'Region', 'y': 'Total Storage (GB)'},
            color=s3_region_storage.values,
            color_continuous_scale='Greens'
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Resource distribution
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🖥️ EC2 Instance Types")
        instance_counts = ec2_df['InstanceType'].value_counts()
        fig = px.pie(
            values=instance_counts.values,
            names=instance_counts.index,
            title="Distribution of Instance Types"
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 💾 S3 Storage Classes")
        storage_counts = s3_df['StorageClass'].value_counts()
        fig = px.pie(
            values=storage_counts.values,
            names=storage_counts.index,
            title="Distribution of Storage Classes"
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

def show_ec2_analysis(ec2_df):
    """Display EC2 detailed analysis"""
    st.markdown("## 🖥️ EC2 Compute Analysis")
    
    # Filters
    st.sidebar.markdown("### 🔍 Filters")
    selected_regions = st.sidebar.multiselect(
        "Select Regions:",
        options=ec2_df['Region'].unique(),
        default=ec2_df['Region'].unique()
    )
    
    selected_states = st.sidebar.multiselect(
        "Select States:",
        options=ec2_df['State'].unique(),
        default=ec2_df['State'].unique()
    )
    
    # Filter data
    filtered_df = ec2_df[
        (ec2_df['Region'].isin(selected_regions)) &
        (ec2_df['State'].isin(selected_states))
    ]
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Instances", len(filtered_df))
    with col2:
        st.metric("Total Cost/Hour", f"${filtered_df['CostUSD'].sum():.2f}")
    with col3:
        st.metric("Avg CPU %", f"{filtered_df['CPUUtilization'].mean():.1f}%")
    with col4:
        st.metric("Avg Memory %", f"{filtered_df['MemoryUtilization'].mean():.1f}%")
    
    st.markdown("---")
    
    # CPU Utilization Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 CPU Utilization Distribution")
        fig = px.histogram(
            filtered_df,
            x='CPUUtilization',
            nbins=30,
            labels={'CPUUtilization': 'CPU Utilization (%)'},
            color_discrete_sequence=['#3498db']
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Memory Utilization Distribution")
        fig = px.histogram(
            filtered_df,
            x='MemoryUtilization',
            nbins=30,
            labels={'MemoryUtilization': 'Memory Utilization (%)'},
            color_discrete_sequence=['#e74c3c']
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # CPU vs Cost Scatter
    st.markdown("### 💰 CPU Utilization vs Cost Analysis")
    fig = px.scatter(
        filtered_df,
        x='CPUUtilization',
        y='CostUSD',
        color='InstanceType',
        size='MemoryUtilization',
        hover_data=['ResourceId', 'Region', 'State'],
        labels={
            'CPUUtilization': 'CPU Utilization (%)',
            'CostUSD': 'Cost per Hour (USD)',
            'MemoryUtilization': 'Memory %'
        }
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Top expensive instances
    st.markdown("### 💸 Top 5 Most Expensive Instances")
    top_expensive = filtered_df.nlargest(5, 'CostUSD')[
        ['ResourceId', 'InstanceType', 'Region', 'State', 'CostUSD', 'CPUUtilization', 'MemoryUtilization']
    ]
    st.dataframe(top_expensive, use_container_width=True)
    
    # Cost by region and instance type
    st.markdown("### 📊 Cost Analysis by Region and Instance Type")
    cost_pivot = filtered_df.pivot_table(
        values='CostUSD',
        index='Region',
        columns='InstanceType',
        aggfunc='sum',
        fill_value=0
    )
    fig = px.imshow(
        cost_pivot,
        labels=dict(x="Instance Type", y="Region", color="Cost (USD)"),
        aspect="auto",
        color_continuous_scale='YlOrRd'
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

def show_s3_analysis(s3_df):
    """Display S3 detailed analysis"""
    st.markdown("## 💾 S3 Storage Analysis")
    
    # Filters
    st.sidebar.markdown("### 🔍 Filters")
    selected_regions = st.sidebar.multiselect(
        "Select Regions:",
        options=s3_df['Region'].unique(),
        default=s3_df['Region'].unique()
    )
    
    selected_storage_class = st.sidebar.multiselect(
        "Select Storage Class:",
        options=s3_df['StorageClass'].unique(),
        default=s3_df['StorageClass'].unique()
    )
    
    # Filter data
    filtered_df = s3_df[
        (s3_df['Region'].isin(selected_regions)) &
        (s3_df['StorageClass'].isin(selected_storage_class))
    ]
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Buckets", len(filtered_df))
    with col2:
        st.metric("Total Storage", f"{filtered_df['TotalSizeGB'].sum():.0f} GB")
    with col3:
        st.metric("Total Cost/Month", f"${filtered_df['CostUSD'].sum():.2f}")
    with col4:
        st.metric("Total Objects", f"{filtered_df['ObjectCount'].sum():,.0f}")
    
    st.markdown("---")
    
    # Storage by region
    st.markdown("### 🌍 Total Storage by Region")
    storage_by_region = filtered_df.groupby('Region')['TotalSizeGB'].sum().sort_values(ascending=False)
    fig = px.bar(
        x=storage_by_region.index,
        y=storage_by_region.values,
        labels={'x': 'Region', 'y': 'Total Storage (GB)'},
        color=storage_by_region.values,
        color_continuous_scale='Viridis'
    )
    fig.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Storage vs Cost
    st.markdown("### 💰 Storage Size vs Cost Analysis")
    fig = px.scatter(
        filtered_df,
        x='TotalSizeGB',
        y='CostUSD',
        color='StorageClass',
        size='ObjectCount',
        hover_data=['BucketName', 'Region', 'Encryption'],
        labels={
            'TotalSizeGB': 'Total Storage (GB)',
            'CostUSD': 'Monthly Cost (USD)',
            'ObjectCount': 'Object Count'
        }
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Top 5 largest buckets
    st.markdown("### 📦 Top 5 Largest S3 Buckets")
    top_large = filtered_df.nlargest(5, 'TotalSizeGB')[
        ['BucketName', 'Region', 'TotalSizeGB', 'CostUSD', 'StorageClass', 'ObjectCount', 'Encryption']
    ]
    st.dataframe(top_large, use_container_width=True)
    
    # Storage class comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 💾 Storage by Class")
        storage_class_data = filtered_df.groupby('StorageClass')['TotalSizeGB'].sum()
        fig = px.pie(
            values=storage_class_data.values,
            names=storage_class_data.index,
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🔒 Encryption Status")
        encryption_data = filtered_df['Encryption'].value_counts()
        fig = px.pie(
            values=encryption_data.values,
            names=encryption_data.index,
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Cost efficiency by storage class
    st.markdown("### 💵 Cost Efficiency by Storage Class")
    cost_efficiency = filtered_df.groupby('StorageClass').agg({
        'TotalSizeGB': 'sum',
        'CostUSD': 'sum'
    }).reset_index()
    cost_efficiency['Cost per GB'] = cost_efficiency['CostUSD'] / cost_efficiency['TotalSizeGB']
    
    fig = px.bar(
        cost_efficiency,
        x='StorageClass',
        y='Cost per GB',
        color='Cost per GB',
        color_continuous_scale='RdYlGn_r',
        labels={'Cost per GB': 'Cost per GB (USD)'}
    )
    st.plotly_chart(fig, use_container_width=True)

def show_optimization(ec2_df, s3_df):
    """Display optimization recommendations"""
    st.markdown("## 🔧 Optimization Recommendations")
    
    # EC2 Optimizations
    st.markdown("### 🖥️ EC2 Optimization Opportunities")
    
    # Low CPU utilization
    low_cpu_threshold = st.slider("CPU Utilization Threshold (%)", 0, 50, 20)
    low_cpu = ec2_df[ec2_df['CPUUtilization'] < low_cpu_threshold]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📉 Underutilized Instances")
        st.metric(
            "Instances with Low CPU",
            len(low_cpu),
            delta=f"<{low_cpu_threshold}% CPU"
        )
        st.metric(
            "Current Hourly Cost",
            f"${low_cpu['CostUSD'].sum():.2f}"
        )
        st.metric(
            "Potential Savings (50% reduction)",
            f"${low_cpu['CostUSD'].sum() * 0.5:.2f}/hour",
            delta=f"${low_cpu['CostUSD'].sum() * 0.5 * 730:.2f}/month"
        )
    
    with col2:
        st.markdown("#### 💡 Recommendation")
        st.info("""
        **Action:** Right-size these instances
        
        - Downgrade to smaller instance types
        - Use AWS Compute Optimizer for recommendations
        - Consider using Auto Scaling
        - Review workload patterns
        """)
    
    if len(low_cpu) > 0:
        st.markdown("##### Underutilized Instances Details")
        st.dataframe(
            low_cpu[['ResourceId', 'InstanceType', 'Region', 'CPUUtilization', 'CostUSD', 'State']].head(10),
            use_container_width=True
        )
    
    st.markdown("---")
    
    # Stopped instances
    stopped = ec2_df[ec2_df['State'] == 'stopped']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ⏸️ Stopped Instances")
        st.metric("Stopped Instances", len(stopped))
        st.warning("Stopped instances still incur EBS storage costs!")
    
    with col2:
        st.markdown("#### 💡 Recommendation")
        st.info("""
        **Action:** Review stopped instances
        
        - Terminate unused instances
        - Create AMIs for backup
        - Use snapshots instead of keeping instances
        - Set up automated cleanup policies
        """)
    
    st.markdown("---")
    
    # S3 Optimizations
    st.markdown("### 💾 S3 Optimization Opportunities")
    
    # Storage class optimization
    standard_buckets = s3_df[s3_df['StorageClass'] == 'STANDARD']
    large_standard = standard_buckets[standard_buckets['TotalSizeGB'] > 1000]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📦 Storage Class Optimization")
        st.metric(
            "Large STANDARD Buckets (>1TB)",
            len(large_standard)
        )
        st.metric(
            "Total Size",
            f"{large_standard['TotalSizeGB'].sum():.0f} GB"
        )
        st.metric(
            "Current Monthly Cost",
            f"${large_standard['CostUSD'].sum():.2f}"
        )
        st.metric(
            "Potential Savings (60% reduction)",
            f"${large_standard['CostUSD'].sum() * 0.6:.2f}/month",
            delta="Move to GLACIER/STANDARD_IA"
        )
    
    with col2:
        st.markdown("#### 💡 Recommendation")
        st.info("""
        **Action:** Optimize storage classes
        
        - Move infrequently accessed data to STANDARD_IA
        - Archive old data to GLACIER
        - Use S3 Lifecycle policies
        - Enable S3 Intelligent-Tiering
        """)
    
    if len(large_standard) > 0:
        st.markdown("##### Large STANDARD Buckets Details")
        st.dataframe(
            large_standard[['BucketName', 'Region', 'TotalSizeGB', 'CostUSD', 'ObjectCount']].head(10),
            use_container_width=True
        )
    
    st.markdown("---")
    
    # Unencrypted buckets
    no_encryption = s3_df[s3_df['Encryption'] == 'None']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔓 Unencrypted Buckets")
        st.metric("Buckets without Encryption", len(no_encryption))
        st.metric("Total Size", f"{no_encryption['TotalSizeGB'].sum():.0f} GB")
        st.error("Security risk! Enable encryption for compliance.")
    
    with col2:
        st.markdown("#### 💡 Recommendation")
        st.info("""
        **Action:** Enable encryption
        
        - Enable AES256 encryption (no additional cost)
        - Use AWS KMS for key management
        - Set default encryption on buckets
        - Audit access policies
        """)
    
    # Summary
    st.markdown("---")
    st.markdown("### 📊 Optimization Summary")
    
    total_ec2_savings = low_cpu['CostUSD'].sum() * 0.5 * 730
    total_s3_savings = large_standard['CostUSD'].sum() * 0.6
    total_savings = total_ec2_savings + total_s3_savings
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "💰 Total Monthly Savings Potential",
            f"${total_savings:.2f}",
            delta="Estimated"
        )
    
    with col2:
        st.metric(
            "🖥️ EC2 Savings",
            f"${total_ec2_savings:.2f}",
            delta="Right-sizing"
        )
    
    with col3:
        st.metric(
            "💾 S3 Savings",
            f"${total_s3_savings:.2f}",
            delta="Storage optimization"
        )

if __name__ == "__main__":
    main()