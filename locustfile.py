import random
from locust import HttpUser, task, between

# Sample financial questions for load testing
QUESTIONS = [
    "What was Berkshire Hathaway's total revenue in 2024?",
    "What was Berkshire's net earnings in 2023?",
    "Who is the CEO of Berkshire Hathaway?",
    "What are Berkshire's main business segments?",
    "What cybersecurity risks does Berkshire mention in 2024?",
    "What was BNSF's revenue in 2022?",
    "What is Berkshire's investment in Apple in 2024?",
    "How does Berkshire make money from insurance?",
    "What was Berkshire's cash position in 2021?",
    "What are the main risks to BNSF railroad?",
    "What was GEICO's revenue in 2023?",
    "What was Berkshire's book value per share in 2024?",
    "What is Berkshire's approach to capital allocation?",
    "What was Berkshire's operating earnings in 2020?",
    "What climate risks does Berkshire face in 2024?",
]

class RAGUser(HttpUser):
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests

    @task(3)
    def query_auto(self):
        """Most common — auto routing"""
        question = random.choice(QUESTIONS)
        self.client.post(
            "/query",
            json={"question": question, "strategy": "auto"},
            name="/query (auto)"
        )

    @task(1)
    def query_strategy1(self):
        """Explicit strategy 1"""
        question = random.choice(QUESTIONS)
        self.client.post(
            "/query",
            json={"question": question, "strategy": "strategy1"},
            name="/query (strategy1)"
        )

    @task(1)
    def check_metrics(self):
        """Check metrics endpoint"""
        self.client.get("/metrics", name="/metrics")

    @task(1)
    def check_logs(self):
        """Check logs endpoint"""
        self.client.get("/logs", name="/logs")