import random
from typing import List, Dict

class ITTicketDataset:
    def __init__(self):
        self.issue_types = ["Hardware", "Software", "Network", "Security", "Access"]
        self.priorities = ["Low", "Medium", "High", "Critical"]
        self.statuses = ["Open", "In Progress", "Resolved", "Closed"]
        
    def generate_description(self, issue_type: str) -> str:
        descriptions = {
            "Hardware": [
                "Computer won't start up",
                "Printer not responding",
                "Monitor display issues",
                "Keyboard malfunction",
                "Mouse not working"
            ],
            "Software": [
                "Application crashes frequently",
                "Software update failed",
                "Program not responding",
                "Error message on startup",
                "Data sync issues"
            ],
            "Network": [ 
                "Internet connection down",
                "Slow network performance",
                "Cannot access shared drive",
                "VPN connection issues",
                "Wi-Fi connectivity problems"
            ],
            "Security": [
                "Suspicious email received",
                "Account locked out",
                "Potential malware detected",
                "Password reset required",
                "Unauthorized access attempt"
            ],
            "Access": [
                "Cannot log into system",
                "Permission denied error",
                "New user setup required",
                "Account activation needed",
                "Resource access request"
            ]
        }
        return random.choice(descriptions[issue_type])

    def generate_ticket(self) -> Dict:
        issue_type = random.choice(self.issue_types)
        return {
            "ticket_id": f"TIC-{random.randint(1000, 9999)}",
            "issue_type": issue_type,
            "description": self.generate_description(issue_type),
            "priority": random.choice(self.priorities),
            "status": random.choice(self.statuses)
        }

    def generate_dataset(self, num_tickets: int = 100) -> List[Dict]:
        tickets = [self.generate_ticket() for _ in range(num_tickets)]
        return tickets
