# src/utils/alerting.py
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging
from datetime import datetime

class AlertSystem:
    def __init__(self, smtp_server=None, smtp_port=587):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.logger = logging.getLogger(__name__)
    
    def send_email_alert(self, to_email, subject, message, from_email=None, password=None):
        """Envoie une alerte par email"""
        try:
            msg = MIMEMultipart()
            msg['From'] = from_email
            msg['To'] = to_email
            msg['Subject'] = subject
            
            msg.attach(MIMEText(message, 'plain'))
            
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(from_email, password)
            text = msg.as_string()
            server.sendmail(from_email, to_email, text)
            server.quit()
            
            self.logger.info(f"Alert sent to {to_email}")
        except Exception as e:
            self.logger.error(f"Failed to send alert: {e}")
    
    def log_alert(self, alert_type, message, level="WARNING"):
        """Log une alerte"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        alert_message = f"[{timestamp}] {alert_type}: {message}"
        
        if level == "ERROR":
            self.logger.error(alert_message)
        elif level == "WARNING":
            self.logger.warning(alert_message)
        else:
            self.logger.info(alert_message)
    
    def data_quality_alert(self, issue_type, details):
        """Alerte spécifique pour la qualité des données"""
        message = f"Data Quality Issue Detected: {issue_type}\nDetails: {details}"
        self.log_alert("DATA_QUALITY", message, "ERROR")
    
    def model_performance_alert(self, model_name, metric, threshold, current_value):
        """Alerte pour la performance des modèles"""
        message = f"Model Performance Alert: {model_name}\n"
        message += f"Metric: {metric}\n"
        message += f"Threshold: {threshold}\n"
        message += f"Current Value: {current_value}"
        self.log_alert("MODEL_PERFORMANCE", message, "WARNING")
