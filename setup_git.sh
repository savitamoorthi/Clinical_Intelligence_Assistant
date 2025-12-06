#!/bin/bash

# =============================================================================
# Quick Git Setup Commands for Clinical Intelligence Assistant
# Copy and paste these commands in order
# =============================================================================

echo "🚀 Starting Git Setup for Clinical Intelligence Assistant"
echo ""

# Navigate to project directory
cd /Users/savitamoorthi/Desktop/Clinical_Intelligence_Assistant

# Step 1: Initialize Git
echo "📦 Step 1: Initializing Git repository..."
git init
echo "✅ Git initialized"
echo ""

# Step 2: Check .gitignore is working
echo "🔒 Step 2: Verifying .gitignore is protecting secrets..."
echo "Checking if .env will be ignored..."
git check-ignore .env
if [ $? -eq 0 ]; then
    echo "✅ .env file will be ignored (GOOD!)"
else
    echo "⚠️  WARNING: .env might not be ignored!"
fi
echo ""

# Step 3: Add all files
echo "📝 Step 3: Staging files..."
git add .
echo "✅ Files staged"
echo ""

# Step 4: Show what will be committed
echo "👀 Step 4: Review files to be committed..."
echo "IMPORTANT: Make sure .env is NOT in this list!"
echo "─────────────────────────────────────────"
git status
echo "─────────────────────────────────────────"
echo ""
echo "⚠️  STOP HERE AND VERIFY:"
echo "   1. .env is NOT listed above"
echo "   2. node_modules/ is NOT listed above"
echo "   3. __pycache__/ is NOT listed above"
echo ""
read -p "Do you see .env in the list above? (y/n): " response

if [[ "$response" == "y" || "$response" == "Y" ]]; then
    echo "❌ ERROR: .env should not be committed!"
    echo "Run: git rm --cached .env"
    exit 1
else
    echo "✅ Looks good! Continuing..."
fi
echo ""

# Step 5: Create initial commit
echo "💾 Step 5: Creating initial commit..."
git commit -m "Initial commit: Clinical Intelligence Assistant - UChicago MSADS Capstone

Multi-agent AI system for clinical intelligence with:
- EHR Agent (Text-to-SQL)
- Reports Agent (RAG)
- Supervisor Agent (Orchestration)
- Full-stack React + FastAPI application
- RAGAS evaluation framework"
echo "✅ Commit created"
echo ""

# Step 6: Instructions for GitHub
echo "🌐 Step 6: Next steps for GitHub..."
echo ""
echo "Now you need to:"
echo "1. Go to: https://github.com/new"
echo "2. Repository name: Clinical_Intelligence_Assistant"
echo "3. Description: Multi-Agent Clinical Intelligence System - MS Applied Data Science Capstone @ UChicago"
echo "4. Select Public (recommended for portfolio) or Private" 
echo "5. DO NOT initialize with README, .gitignore, or license"
echo "6. Click 'Create repository'"
echo ""
echo "After creating the repository, GitHub will show you a URL like:"
echo "https://github.com/savitamoorthi/Clinical_Intelligence_Assistant.git"
echo ""
read -p "Have you created the repository? Press Enter when ready to continue..."
echo ""

# Step 7: Add remote (you'll need to edit this with your actual URL)
echo "📡 Step 7: Add GitHub remote..."
echo "Paste the repository URL from GitHub (e.g., https://github.com/savitamoorthi/Clinical_Intelligence_Assistant.git):"
read repo_url

git remote add origin "$repo_url"
echo "✅ Remote added"
echo ""

# Step 8: Verify remote
echo "🔍 Step 8: Verifying remote..."
git remote -v
echo ""

# Step 9: Rename branch to main (if needed)
echo "🌿 Step 9: Ensuring branch is named 'main'..."
git branch -M main
echo "✅ Branch renamed to main"
echo ""

# Step 10: Push to GitHub
echo "⬆️  Step 10: Pushing to GitHub..."
echo "You may be prompted for credentials..."
git push -u origin main
echo ""

# Final check
if [ $? -eq 0 ]; then
    echo "═══════════════════════════════════════════════════════════"
    echo "✨ SUCCESS! Your project is now on GitHub! ✨"
    echo "═══════════════════════════════════════════════════════════"
    echo ""
    echo "View your repository at:"
    echo "https://github.com/savitamoorthi/Clinical_Intelligence_Assistant"
    echo ""
    echo "Next steps:"
    echo "1. Review your repository online"
    echo "2. VERIFY .env is NOT visible"
    echo "3. Consider adding topics: machine-learning, healthcare, multi-agent-systems, langchain"
    echo "4. Star your own repo 😊"
    echo ""
else
    echo "❌ Push failed. Check the error message above."
    echo "Common issues:"
    echo "- Authentication: You may need a Personal Access Token"
    echo "- Repository URL: Verify the URL is correct"
fi
