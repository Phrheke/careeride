from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_sqlalchemy import SQLAlchemy
import numpy as np
import pickle
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Load the trained model for recommendation
model = pickle.load(open('career_recommendation_model.pkl', 'rb'))

# Define the list of interests for recommendation
interests = ["Coding", "Data Analysis", "Cybersecurity", "AI/ML", "Web Development",
             "Database Management", "Cloud Computing", "Software Testing", "Version Control",
             "Network Security", "Ethical Hacking", "Cryptography", "System Administration",
             "Linux", "Windows Administration", "Mobile App Development", "UI/UX Design",
             "Frontend Development", "Backend Development", "Full-Stack Development",
             "Natural Language Processing", "Computer Vision", "Data Visualization",
             "Big Data", "Machine Learning", "Deep Learning", "Data Mining",
             "Business Intelligence", "Statistics", "Mathematics", "Algorithms",
             "Data Structures", "Agile Methodologies", "Project Management", "DevOps", "CI/CD",
             "API Development", "Containerization", "Kubernetes", "Virtualization",
             "Cloud Architecture", "AWS", "Azure", "Google Cloud", "Blockchain",
             "Quantum Computing", "Artificial Intelligence", "Problem-Solving"]

# Define the questionnaire questions and their impact on careers
questions = [
    {'question': 'Are you proficient in programming languages such as Java, Python, C++, or JavaScript?', 'careers': {'Software Developer': 15}},
    {'question': 'Are you comfortable with solving complex coding challenges or algorithmic problems?', 'careers': {'Software Developer': 15}},
    {'question': 'Do you have experience with frameworks like React, Angular, Spring, or Django?', 'careers': {'Software Developer': 15}},
    {'question': 'Do you use software development tools like VS Code, IntelliJ, or Git often?', 'careers': {'Software Developer': 10}},
    {'question': 'Do you have a strong understanding of statistical analysis, probability, and mathematical modeling?', 'careers': {'Data Scientist': 15}},
    {'question': 'Are you proficient in programming languages like Python, R, and SQL for data analysis?', 'careers': {'Data Scientist': 15}},
    {'question': 'Are you familiar with machine learning algorithms and tools like TensorFlow or Scikit-Learn?', 'careers': {'Data Scientist': 15}},
    {'question': 'Are you comfortable with using data manipulation and cleaning using tools like Pandas or Excel?', 'careers': {'Data Scientist': 10}},
    {'question': 'Are you familiar with security protocols, encryption, firewalls, and intrusion detection systems?', 'careers': {'Cybersecurity Analyst': 20}},
    {'question': 'Are you experienced in ethical hacking and penetration testing, including tools like Metasploit?', 'careers': {'Cybersecurity Analyst': 20}},
    {'question': 'Do you have a strong understanding of network security, including protocols like TCP/IP and DNS?', 'careers': {'Cybersecurity Analyst': 15}},
    {'question': 'Are you knowledgeable about risk management and compliance with regulations like GDPR or HIPAA?', 'careers': {'Cybersecurity Analyst': 10}},
    {'question': 'Are you familiar with setting up and managing CI/CD pipelines using tools like Jenkins or GitLab CI?', 'careers': {'DevOps Engineer': 15}},
    {'question': 'Are you proficient in using Infrastructure as Code (IaC) tools like Terraform or Ansible?', 'careers': {'DevOps Engineer': 15}},
    {'question': 'Are you experienced with cloud platforms like AWS, Azure, or Google Cloud for deploying applications?', 'careers': {'DevOps Engineer': 15}},
    {'question': 'Are you comfortable with scripting and automation using languages like Bash, Python, or PowerShell?', 'careers': {'DevOps Engineer': 10}},
    {'question': 'How deep is your understanding of machine learning algorithms and tools like TensorFlow or PyTorch?', 'careers': {'AI/ML Engineer': 15}},
    {'question': 'Are you proficient in programming languages like Python or R, specifically for AI/ML development?', 'careers': {'AI/ML Engineer': 15}},
    {'question': 'Have you used tools like Pandas or NumPy for data manipulation and cleaning?', 'careers': {'AI/ML Engineer': 15}},
    {'question': 'Do you have a strong foundation in mathematics, including linear algebra, calculus, and statistics?', 'careers': {'AI/ML Engineer': 10}},
    {'question': 'Are you proficient in front-end development using HTML, CSS, JavaScript, and frameworks like React or Angular?', 'careers': {'Web Developer': 15}},
    {'question': 'Do you have experience in back-end development with server-side languages like Node.js, Python, or Ruby on Rails?', 'careers': {'Web Developer': 15}},
    {'question': 'Are you familiar with databases, both relational (e.g., MySQL) and NoSQL (e.g., MongoDB)?', 'careers': {'Web Developer': 15}},
    {'question': 'Are you familiar with API development and integration, especially RESTful APIs?', 'careers': {'Web Developer': 10}},
    {'question': 'Are you proficient in mobile programming languages like Swift, Kotlin, Java, or cross-platform tools like Flutter?', 'careers': {'Mobile App Developer': 15}},
    {'question': 'Are you skilled in UI/UX design specifically for mobile applications?', 'careers': {'Mobile App Developer': 20}},
    {'question': 'Are you familiar with mobile app frameworks like Android Studio or Xcode?', 'careers': {'Mobile App Developer': 15}},
    {'question': 'Can you integrate mobile apps with backend services using APIs?', 'careers': {'Mobile App Developer': 10}},
    {'question': 'Do you pay attention to detail when working on technical tasks?', 'careers': {'Software Developer': 10, 'Data Scientist': 5, 'Cybersecurity Analyst': 10, 'DevOps Engineer': 5, 'AI/ML Engineer': 10, 'Web Developer': 10, 'Mobile App Developer': 10}},
    {'question': 'Are you comfortable working in a team and collaborating with others?', 'careers': {'Software Developer': 5, 'Data Scientist': 5, 'Cybersecurity Analyst': 5, 'DevOps Engineer': 10, 'AI/ML Engineer': 5, 'Web Developer': 10, 'Mobile App Developer': 5}},
    {'question': 'Are you patient and persistent when solving challenging problems?', 'careers': {'Software Developer': 15, 'Data Scientist': 10, 'Cybersecurity Analyst': 10, 'DevOps Engineer': 5, 'AI/ML Engineer': 10, 'Web Developer': 5, 'Mobile App Developer': 10}},
    {'question': 'Do you have effective communication skills in explaining technical concepts?', 'careers': {'Software Developer': 5, 'Data Scientist': 10, 'Cybersecurity Analyst': 5, 'DevOps Engineer': 10, 'AI/ML Engineer': 5, 'Web Developer': 5, 'Mobile App Developer': 5}},
    {'question': 'Do you have strong problem-solving abilities when faced with complex technical issues?', 'careers': {'Software Developer': 10, 'Data Scientist': 15, 'Cybersecurity Analyst': 5, 'DevOps Engineer': 15, 'AI/ML Engineer': 15, 'Web Developer': 15, 'Mobile App Developer': 10}}
]

responses = {
    'Yes': 1.0,
    'Neutral': 0.5,
    'No': 0.0
}

def calculate_scores(answers):
    career_scores = {
        'Software Developer': 0,
        'Data Scientist': 0,
        'Cybersecurity Analyst': 0,
        'DevOps Engineer': 0,
        'AI/ML Engineer': 0,
        'Web Developer': 0,
        'Mobile App Developer': 0
    }
    
    for i, q in enumerate(questions):
        user_response = answers[i]
        impact = responses[user_response]
        for career, percentage in q['careers'].items():
            career_scores[career] += percentage * impact

    sorted_careers = sorted(career_scores.items(), key=lambda item: item[1], reverse=True)
    top_two_careers = sorted_careers[:2]
    return top_two_careers

# Define database model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)
    recommendations = db.relationship('Recommendation', backref='user', lazy=True)

class Recommendation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    career = db.Column(db.String(100), nullable=False)
    date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

def create_db():
    with app.app_context():
        db.create_all()
@app.route('/')
def home():
    return redirect(url_for('login'))  # Redirect to the login page or another page

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        
        # Check if user already exists
        if User.query.filter_by(email=email).first():
            return 'User already exists!'
        
        user = User(name=name, email=email, password=password)
        db.session.add(user)
        db.session.commit()
        
        session['user_id'] = user.id
        return redirect(url_for('dashboard'))
    
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        user = User.query.filter_by(email=email, password=password).first()
        
        if user:
            session['user_id'] = user.id
            return redirect(url_for('dashboard'))
        else:
            return 'Invalid email or password'
    
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    user = User.query.get(user_id)
    recommendations = Recommendation.query.filter_by(user_id=user_id).all()
    
    return render_template('dashboard.html', name=user.name, recommendations=recommendations, interests=interests)

@app.route('/recommend', methods=['POST'])
def recommend():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    data = request.json
    selected_interests = data['interest']
    
    input_vector = [1 if interest in selected_interests else 0 for interest in interests]
    prediction = model.predict([input_vector])[0]
    
    user_id = session['user_id']
    new_recommendation = Recommendation(career=prediction, user_id=user_id)
    db.session.add(new_recommendation)
    db.session.commit()
    
    return jsonify({'message': prediction})

@app.route('/submit_questionnaire', methods=['POST'])
def submit_questionnaire():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    answers = request.json['answers']
    top_careers = calculate_scores(answers)
    
    user_id = session['user_id']
    for career in top_careers:
        new_recommendation = Recommendation(career=career[0], user_id=user_id)
        db.session.add(new_recommendation)
    db.session.commit()
    
    return jsonify({'careers': top_careers})

@app.route('/questionnaire')
def questionnaire():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    return render_template('questionnaire.html', questions=questions)

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    create_db()  # Create the database and tables
    app.run(debug=True)
