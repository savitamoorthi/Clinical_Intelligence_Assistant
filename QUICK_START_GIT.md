# 🚀 Quick Start: Push to GitHub

## ⚡ Option 1: Run the Interactive Script (RECOMMENDED)

```bash
cd /Users/savitamoorthi/Desktop/Clinical_Intelligence_Assistant
./setup_git.sh
```

This script will guide you through each step with safety checks!

---

## ⚡ Option 2: Manual Steps (Copy & Paste Each Command)

### 1️⃣ Create GitHub Repository First!

**Go to:** https://github.com/new

- **Repository name:** `Clinical_Intelligence_Assistant`
- **Description:** Multi-Agent Clinical Intelligence System - MS Applied Data Science Capstone @ UChicago
- **Public or Private:** Your choice (Public recommended for portfolio)
- **DO NOT check:** Add README, .gitignore, or license
- **Click:** "Create repository"

---

### 2️⃣ Initialize Git Locally

```bash
cd /Users/savitamoorthi/Desktop/Clinical_Intelligence_Assistant
git init
```

---

### 3️⃣ CRITICAL SECURITY CHECK ✋

Verify .env will be ignored:

```bash
git check-ignore .env
```

**Expected output:** `.env` (means it WILL be ignored - this is GOOD!)

If you get no output, STOP and run:
```bash
echo ".env" >> .gitignore
```

---

### 4️⃣ Stage Files

```bash
git add .
```

---

### 5️⃣ SECOND SECURITY CHECK ✋

```bash
git status
```

**VERIFY:** `.env` is **NOT** in the list of files to be committed!

If you see `.env` in the list, STOP and run:
```bash
git rm --cached .env
```

---

### 6️⃣ Create Initial Commit

```bash
git commit -m "Initial commit: Clinical Intelligence Assistant - UChicago MSADS Capstone

Multi-agent AI system for clinical intelligence with:
- EHR Agent (Text-to-SQL)
- Reports Agent (RAG)
- Supervisor Agent (Orchestration)
- Full-stack React + FastAPI application
- RAGAS evaluation framework"
```

---

### 7️⃣ Connect to GitHub

**Replace `YOUR_REPO_URL` with the URL from step 1:**

```bash
git remote add origin https://github.com/savitamoorthi/Clinical_Intelligence_Assistant.git
```

Verify:
```bash
git remote -v
```

---

### 8️⃣ Rename Branch to Main

```bash
git branch -M main
```

---

### 9️⃣ Push to GitHub

```bash
git push -u origin main
```

**Note:** You may need to enter your GitHub credentials or Personal Access Token.

---

### 🔟 FINAL VERIFICATION ✅

1. Go to: `https://github.com/savitamoorthi/Clinical_Intelligence_Assistant`
2. **CRITICAL CHECK:** Verify `.env` is **NOT** visible
3. Verify `README.md` displays correctly
4. Check that all your code files are there

---

## 🔐 Getting a GitHub Personal Access Token (if needed)

If `git push` asks for authentication:

1. Go to: https://github.com/settings/tokens
2. Click: "Generate new token" → "Generate new token (classic)"
3. Give it a name: "Clinical Intelligence Assistant"
4. Select scope: ✅ `repo` (full control of private repositories)
5. Click: "Generate token"
6. **COPY THE TOKEN** (you won't see it again!)
7. Use this token as your **password** when prompted during `git push`

---

## 📝 Future Commits (After Initial Setup)

```bash
# Check what changed
git status

# Add changes
git add .

# Commit
git commit -m "Describe your changes here"

# Push to GitHub
git push
```

---

## 🆘 Troubleshooting

### ❌ "Permission denied"
- Use HTTPS URL: `https://github.com/savitamoorthi/Clinical_Intelligence_Assistant.git`
- Get a Personal Access Token (see above)

### ❌ "Repository not found"
- Double-check repository URL
- Make sure you created the repository on GitHub first

### ❌ Accidentally committed .env
1. **IMMEDIATELY revoke all API keys!**
   - OpenAI: https://platform.openai.com/api-keys
   - Google: https://console.cloud.google.com/apis/credentials
2. Run:
   ```bash
   git rm --cached .env
   git commit -m "Remove .env from tracking"
   git push
   ```
3. Generate new API keys

---

## ✅ Success Checklist

After pushing, verify:

- [ ] Repository is visible on GitHub
- [ ] README.md displays correctly
- [ ] `.env` is **NOT** visible in repository
- [ ] `.gitignore` is present
- [ ] All code files are present
- [ ] node_modules/ is **NOT** in repository
- [ ] __pycache__/ is **NOT** in repository

---

## 🎯 Recommended: Add Repository Topics

On GitHub, add these topics to your repository for better discoverability:

- `machine-learning`
- `artificial-intelligence`
- `healthcare`
- `clinical-informatics`
- `multi-agent-systems`
- `langchain`
- `langgraph`
- `retrieval-augmented-generation`
- `text-to-sql`
- `capstone-project`
- `university-of-chicago`
- `data-science`

---

## 📧 Questions?

Refer to the detailed `GIT_SETUP_GUIDE.md` for more information.

**Happy coding! 🎓**
