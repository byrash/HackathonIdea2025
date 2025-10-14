.PHONY: help install install-backend install-frontend start start-backend start-frontend stop clean

# Default target
help:
	@echo "\033[1;36m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m"
	@echo "\033[1;35m📚 Fraud Detection System - Available Commands\033[0m"
	@echo "\033[1;36m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m"
	@echo ""
	@echo "\033[1;33m📦 Installation:\033[0m"
	@echo "  \033[1;32mmake install\033[0m        - Install all dependencies (backend + frontend)"
	@echo "  \033[1;32mmake install-backend\033[0m - Install backend dependencies"
	@echo "  \033[1;32mmake install-frontend\033[0m - Install frontend dependencies"
	@echo ""
	@echo "\033[1;33m🚀 Running:\033[0m"
	@echo "  \033[1;32mmake start\033[0m          - Start both backend and frontend"
	@echo "  \033[1;32mmake start-backend\033[0m  - Start backend only"
	@echo "  \033[1;32mmake start-frontend\033[0m - Start frontend only"
	@echo ""
	@echo "\033[1;33m🛠️  Maintenance:\033[0m"
	@echo "  \033[1;32mmake stop\033[0m           - Stop all running processes"
	@echo "  \033[1;32mmake clean\033[0m          - Clean up generated files"
	@echo ""
	@echo "\033[1;36m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m"

# Install all dependencies
install: install-backend install-frontend
	@echo "\033[1;32m✓ All dependencies installed!\033[0m"

# Install backend dependencies
install-backend:
	@echo "\033[1;34m📦 Installing backend dependencies...\033[0m"
	cd backend && python -m venv .venv
	cd backend && ./.venv/bin/pip install -r requirements.txt
	@echo "\033[1;32m✓ Backend dependencies installed!\033[0m"

# Install frontend dependencies
install-frontend:
	@echo "\033[1;32m📦 Installing frontend dependencies...\033[0m"
	cd frontend && npm install
	@echo "\033[1;32m✓ Frontend dependencies installed!\033[0m"

# Start both backend and frontend
start: install
	@echo "\033[1;36m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m"
	@echo "\033[1;35m🚀 Starting Fraud Detection System\033[0m"
	@echo "\033[1;36m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m"
	@echo ""
	@echo "\033[1;34m🔵 Backend:\033[0m  http://localhost:8000"
	@echo "\033[1;32m🟢 Frontend:\033[0m http://localhost:4200"
	@echo ""
	@echo "\033[1;33m💡 Press Ctrl+C to stop all services\033[0m"
	@echo "\033[1;36m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m"
	@echo ""
	@trap 'kill 0' EXIT; \
	(cd backend && ./.venv/bin/python main.py 2>&1 | sed 's/^/\x1b[34m[BACKEND]\x1b[0m /') & \
	(cd frontend && npm run serve 2>&1 | sed 's/^/\x1b[32m[FRONTEND]\x1b[0m /') & \
	wait

# Start backend only
start-backend:
	@echo "\033[1;34m🔵 Starting backend on http://localhost:8000...\033[0m"
	cd backend && ./.venv/bin/python main.py

# Start frontend only
start-frontend:
	@echo "\033[1;32m🟢 Starting frontend on http://localhost:4200...\033[0m"
	cd frontend && npm run serve

# Stop all processes (useful if running in background)
stop:
	@echo "\033[1;33m⏹️  Stopping all services...\033[0m"
	@pkill -f "uvicorn main:app" || true
	@pkill -f "vite" || true
	@echo "\033[1;32m✓ All services stopped!\033[0m"

# Clean up generated files
clean:
	@echo "\033[1;33m🧹 Cleaning up...\033[0m"
	rm -rf backend/__pycache__
	rm -rf backend/services/__pycache__
	rm -rf backend/.venv
	rm -rf backend/uploads/*.jpg backend/uploads/*.jpeg backend/uploads/*.png backend/uploads/*.pdf
	rm -rf backend/jobs/*.json
	rm -rf frontend/node_modules
	rm -rf frontend/dist
	@echo "\033[1;32m✓ Cleanup complete!\033[0m"

