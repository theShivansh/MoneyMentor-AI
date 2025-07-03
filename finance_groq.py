import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from groq import Groq

# Set page configuration
st.set_page_config(
    page_title="Personal Finance Tip Generator",
    page_icon="💰",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Initialize Groq client
if 'GROQ_API_KEY' in st.secrets:
    client = Groq(api_key=st.secrets['GROQ_API_KEY'])
else:
    api_key = st.sidebar.text_input("Enter Groq API Key", type="password")
    if not api_key:
        st.info("Please enter your Groq API key to continue")
        st.stop()
    client = Groq(api_key=api_key)

# Custom CSS styling
st.markdown("""
    <style>
    .big-font {
        font-size:24px !important;
        font-weight: bold;
    }
    .highlight {
        background-color: #f0f7ff;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    .tip-box {
        background-color: #e6f7ff;
        border-left: 5px solid #1890ff;
        padding: 10px 15px;
        margin: 15px 0;
        border-radius: 0 5px 5px 0;
    }
    .warning-box {
        background-color: #fff7e6;
        border-left: 5px solid #ffa940;
        padding: 10px 15px;
        margin: 15px 0;
        border-radius: 0 5px 5px 0;
    }
    .success-box {
        background-color: #f6ffed;
        border-left: 5px solid #52c41a;
        padding: 10px 15px;
        margin: 15px 0;
        border-radius: 0 5px 5px 0;
    }
    .expense-chart {
        margin-top: 20px;
        margin-bottom: 30px;
    }
    </style>
    """, unsafe_allow_html=True)

# App title
st.title("💰 Personal Finance Tip Generator")
st.markdown("Get personalized financial advice based on your income, expenses, and savings goals.")
st.markdown("---")

def get_financial_data():
    """Collect financial information via Streamlit form"""
    with st.form("finance_form"):
        st.subheader("Your Financial Details")
        
        # Income section
        st.markdown("#### 💵 Monthly Income")
        income = st.number_input("Total monthly income (₹)", min_value=0.0, step=100.0, value=30000.0)
        
        # Expenses section
        st.markdown("#### 🛒 Monthly Expenses")
        col1, col2 = st.columns(2)
        
        with col1:
            rent = st.number_input("Rent/Mortgage (₹)", min_value=0.0, step=100.0, value=12000.0)
            utilities = st.number_input("Utilities (₹)", min_value=0.0, step=50.0, value=2000.0)
            groceries = st.number_input("Groceries (₹)", min_value=0.0, step=50.0, value=4000.0)
            transport = st.number_input("Transportation (₹)", min_value=0.0, step=50.0, value=3000.0)
            
        with col2:
            dining = st.number_input("Dining Out (₹)", min_value=0.0, step=50.0, value=2000.0)
            entertainment = st.number_input("Entertainment (₹)", min_value=0.0, step=50.0, value=1500.0)
            shopping = st.number_input("Shopping (₹)", min_value=0.0, step=50.0, value=1000.0)
            other = st.number_input("Other Expenses (₹)", min_value=0.0, step=50.0, value=1500.0)
        
        expenses = {
            "Rent/Mortgage": rent,
            "Utilities": utilities,
            "Groceries": groceries,
            "Transportation": transport,
            "Dining Out": dining,
            "Entertainment": entertainment,
            "Shopping": shopping,
            "Other": other
        }
        
        # Savings section
        st.markdown("#### 🏦 Savings")
        savings = st.number_input("Current Savings (₹)", min_value=0.0, step=1000.0, value=5000.0)
        
        # Goals section
        st.markdown("#### 🎯 Savings Goals")
        goal = st.radio("Do you have a savings goal?", ("Yes", "No"), index=1)
        
        target = None
        timeline = None
        if goal == "Yes":
            col1, col2 = st.columns(2)
            with col1:
                target = st.number_input("Target savings amount (₹)", min_value=0.0, step=1000.0, value=10000.0)
            with col2:
                timeline = st.number_input("Timeline (months)", min_value=1, step=1, value=12)
        
        # Submit button
        submitted = st.form_submit_button("Generate Financial Tips")
        
        if submitted:
            return {
                'income': income,
                'expenses': expenses,
                'savings': savings,
                'target': target if goal == "Yes" else None,
                'timeline': timeline if goal == "Yes" else None
            }
    return None

def visualize_expenses(expenses):
    """Create pie chart for expenses"""
    # Clean up zero values
    expenses = {k: v for k, v in expenses.items() if v > 0}
    
    if not expenses:
        return
    
    # Create pie chart
    fig, ax = plt.subplots()
    ax.pie(expenses.values(), labels=expenses.keys(), autopct='%1.1f%%', 
           startangle=90, colors=plt.cm.Pastel1.colors)
    ax.axis('equal')  # Equal aspect ratio ensures pie is drawn as a circle
    ax.set_title('Monthly Expense Breakdown')
    
    # Display in Streamlit
    st.pyplot(fig)

def analyze_finances(data):
    """Generate financial insights using rule-based analysis"""
    tips = []
    total_expenses = sum(data['expenses'].values())
    net_cash = data['income'] - total_expenses
    
    # Rule 1: Basic budget check
    if net_cash < 0:
        tips.append({
            "type": "warning",
            "message": f"🚨 You're spending ₹{abs(net_cash):.2f} more than you earn each month. "
                      "Focus on reducing expenses immediately."
        })
    else:
        savings_rate = (net_cash / data['income']) * 100 if data['income'] > 0 else 0
        if savings_rate < 20:
            tips.append({
                "type": "tip",
                "message": f"💡 Your savings rate is {savings_rate:.1f}% - below the recommended 20%. "
                          "Try to increase savings by reducing discretionary spending."
            })
        else:
            tips.append({
                "type": "success",
                "message": f"✅ Great job! Your savings rate is {savings_rate:.1f}% - "
                          "meeting the recommended 20%."
            })
    
    # Rule 2: Expense analysis
    if data['expenses']:
        largest_category = max(data['expenses'], key=data['expenses'].get)
        tips.append({
            "type": "tip",
            "message": f"🔍 Your largest expense is '{largest_category}' (₹{data['expenses'][largest_category]:.2f}). "
                      "Review this category for potential savings."
        })
    
    # Rule 3: Savings goal check
    if data['target']:
        if data['savings'] >= data['target']:
            tips.append({
                "type": "success",
                "message": f"🎉 Congratulations! You've already reached your savings goal of ₹{data['target']:.2f}."
            })
        else:
            needed = data['target'] - data['savings']
            monthly_needed = needed / data['timeline'] if data['timeline'] > 0 else needed
            if monthly_needed <= net_cash:
                tips.append({
                    "type": "success",
                    "message": f"⏱️ You're on track to reach your goal! Save ₹{monthly_needed:.2f}/month to meet "
                              f"₹{data['target']:.2f} target in {data['timeline']} months."
                })
            else:
                deficit = monthly_needed - net_cash
                tips.append({
                    "type": "warning",
                    "message": f"⚠️ To reach your ₹{data['target']:.2f} goal in {data['timeline']} months, you need to save ₹{monthly_needed:.2f}/month. "
                              f"Current deficit: ₹{deficit:.2f}/month."
                })
    
    # Rule 4: Emergency fund check
    emergency_target = data['income'] * 3
    if data['savings'] < emergency_target:
        tips.append({
            "type": "tip",
            "message": f"🛡️ Consider building an emergency fund (3-6 months of expenses). "
                      f"Aim for at least ₹{emergency_target:.2f} based on your income."
        })
    else:
        tips.append({
            "type": "success",
            "message": f"🛡️✅ Great! Your savings cover at least 3 months of income (emergency fund minimum)."
        })
    
    return tips

def generate_groq_tip(data):
    """Generate personalized tip using Groq API"""
    # Build the savings goal string separately
    savings_goal_str = ""
    if data['target']:
        savings_goal_str = f"- Savings goal: ₹{data['target']:.2f} in {data['timeline']} months\n"
    
    prompt = (
        "As a friendly financial advisor, provide one concise, actionable tip "
        "based on these details:\n"
        f"- Monthly income: ₹{data['income']:.2f}\n"
        f"- Main expenses: {', '.join([f'{k} (₹{v:.2f})' for k, v in data['expenses'].items()])}\n"
        f"- Current savings: ₹{data['savings']:.2f}\n"
        f"{savings_goal_str}\n"
        "Focus on one practical budgeting or saving strategy. "
        "Use plain language and avoid financial jargon. "
        "Keep it to 1-2 sentences maximum. Start with an emoji."
    )
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama3-70b-8192",
            temperature=0.7,
            max_tokens=128,
            top_p=1,
            stop=None,
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Groq API Error: {str(e)}")
        st.error("Please check your API key and internet connection")
        return "🤖 Tip: Review your largest expense category for potential savings."

def main():
    """Main app function"""
    # Sidebar with instructions
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This tool provides personalized financial tips based on:
        - Your monthly income
        - Expense breakdown
        - Current savings
        - Savings goals
        
        **How it works:**
        1. Fill out the financial details
        2. Submit the form
        3. Get rule-based analysis
        4. Receive AI-generated tips using Groq API
        
        All data remains private and is not stored.
        """)
        st.markdown("---")
        st.image("https://console.groq.com/powered-by-groq.svg", width=150)
        st.markdown("Made interactive with Streamlit. Made intelligent with Groq.")
    
    # Get user data
    data = get_financial_data()
    
    if data:
        st.success("Financial data submitted successfully!")
        st.markdown("---")
        
        # Financial summary
        st.subheader("📊 Financial Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Monthly Income", f"₹{data['income']:,.2f}")
        col2.metric("Total Expenses", f"₹{sum(data['expenses'].values()):,.2f}")
        col3.metric("Net Cash Flow", f"₹{data['income'] - sum(data['expenses'].values()):,.2f}")
        
        # Expense visualization
        st.subheader("📈 Expense Breakdown")
        visualize_expenses(data['expenses'])
        
        # Rule-based analysis
        st.subheader("💡 Personalized Finance Tips")
        tips = analyze_finances(data)
        
        for tip in tips:
            if tip["type"] == "tip":
                st.markdown(f'<div class="tip-box">{tip["message"]}</div>', unsafe_allow_html=True)
            elif tip["type"] == "warning":
                st.markdown(f'<div class="warning-box">{tip["message"]}</div>', unsafe_allow_html=True)
            elif tip["type"] == "success":
                st.markdown(f'<div class="success-box">{tip["message"]}</div>', unsafe_allow_html=True)
        
        # Groq-generated tip
        st.subheader("🤖 AI-Powered Advice")
        with st.spinner("Generating personalized financial tip with Groq..."):
            groq_tip = generate_groq_tip(data)
            st.markdown(f'<div class="highlight">{groq_tip}</div>', unsafe_allow_html=True)
        
        # Final notes
        st.markdown("---")
        st.info("💡 Remember: Small consistent changes make big financial differences over time!")

if __name__ == "__main__":
    main()