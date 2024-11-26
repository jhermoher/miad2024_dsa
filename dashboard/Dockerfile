FROM python:3.12

# Create user to run the dashboard
RUN adduser --disabled-password --gecos '' dash-user

# Set working directory
WORKDIR /opt/app

# Copy requirements first and install them
COPY ./app/requirements.txt /opt/app/
COPY ./app/test_requirements.txt /opt/app/
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install -r test_requirements.txt

# Copy the rest of the application
COPY ./app /opt/app/
COPY ./data /opt/app/data/

# Make run script executable
RUN chmod +x /opt/app/run.sh

# Change ownership of the working directory to dash-user
RUN chown -R dash-user:dash-user ./

# Switch to non-root user
USER dash-user

# Expose port for the dashboard
EXPOSE 8050

# Set environment variables
ENV PORT=8050
ENV API_HOST=0.0.0.0
ENV API_PORT=8000

# Run the application
CMD ["bash", "./run.sh"]
