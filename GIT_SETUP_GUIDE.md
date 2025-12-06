# Git Repository Setup Guide

## Steps to Push Your Clinical Intelligence Assistant to GitHub

### Step 1: Create a GitHub Repository

1. Go to [GitHub.com](https://github.com) and sign in with username: **savitamoorthi**
2. Click the **"+"** button in the top-right corner and select **"New repository"**
3. Fill in the repository details:
   - **Repository name**: `Clinical_Intelligence_Assistant` (or your preferred name)
   - **Description**: "Multi-Agent Clinical Intelligence System - MS Applied Data Science Capstone @ UChicago"
   - **Visibility**: Choose **Public** (for academic/portfolio) or **Private** (if you want to control access)
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
4. Click **"Create repository"**

### Step 2: Initialize Git in Your Local Project

Open your terminal and navigate to your project directory:

```bash
cd /Users/savitamoorthi/Desktop/Clinical_Intelligence_Assistant
```

### Step 3: Initialize Git Repository

```bash
# Initialize git repository
git init

# Verify .gitignore is in place (this protects your secrets!)
cat .gitignore
```

### Step 4: Stage Your Files

```bash
# Add all files (respecting .gitignore)
git add .

# Verify what will be committed (make sure .env is NOT listed!)
git status
```

**⚠️ CRITICAL CHECK:** Ensure the following files are **NOT** in the staged files:
- `.env`
- `node_modules/`
- Any files containing API keys

### Step 5: Create Your First Commit

```bash
# Commit your changes
git commit -m "Initial commit: Clinical Intelligence Assistant - UChicago MSADS Capstone"
```

### Step 6: Add Remote Repository

Replace `YOUR_REPO_URL` with the URL from GitHub (should be something like `https://github.com/savitamoorthi/Clinical_Intelligence_Assistant.git`):

```bash
# Add GitHub as remote origin
git remote add origin https://github.com/savitamoorthi/Clinical_Intelligence_Assistant.git

# Verify remote was added
git remote -v
```

### Step 7: Push to GitHub

```bash
# Push to GitHub (use -u to set upstream for future pushes)
git push -u origin main
```

**Note:** If you get an error about the branch name, GitHub might use `master` instead of `main`. In that case:

```bash
# Rename branch to main (if needed)
git branch -M main

# Then push
git push -u origin main
```

### Step 8: Verify Your Repository

1. Go to `https://github.com/savitamoorthi/Clinical_Intelligence_Assistant`
2. Verify all files are there
3. **MOST IMPORTANT:** Check that `.env` is **NOT** visible in the repository

---

## Future Git Commands

### Making Updates

After making changes to your code:

```bash
# Check what changed
git status

# Stage specific files
git add <filename>

# Or stage all changes
git add .

# Commit with descriptive message
git commit -m "Descriptive message about what you changed"

# Push to GitHub
git push
```

### Best Practices for Commit Messages

```bash
# Good examples:
git commit -m "Add RAGAS evaluation for hybrid queries"
git commit -m "Fix SQL injection vulnerability in EHR agent"
git commit -m "Update README with installation instructions"

# Bad examples (avoid):
git commit -m "updates"
git commit -m "fix"
git commit -m "asdf"
```

### Creating Branches (Optional, for feature development)

```bash
# Create and switch to new branch
git checkout -b feature/add-new-agent

# Make your changes, then commit
git add .
git commit -m "Add new specialized agent"

# Push branch to GitHub
git push -u origin feature/add-new-agent

# Then create a Pull Request on GitHub to merge into main
```

### Pulling Latest Changes (if working on multiple machines)

```bash
# Get latest changes from GitHub
git pull origin main
```

---

## Emergency: If You Accidentally Committed Secrets

If you accidentally committed your `.env` file with API keys:

### Option 1: Remove from last commit (if not yet pushed)

```bash
# Remove .env from staging
git rm --cached .env

# Amend the commit
git commit --amend

# Verify .env is in .gitignore
echo ".env" >> .gitignore

# Add gitignore changes
git add .gitignore
git commit -m "Add .env to gitignore"
```

### Option 2: If already pushed to GitHub

**⚠️ CRITICAL:** You must **immediately revoke** all API keys that were exposed!

1. **Revoke API keys immediately:**
   - OpenAI: https://platform.openai.com/api-keys
   - Google: https://console.cloud.google.com/apis/credentials
   - Zilliz: https://cloud.zilliz.com/

2. **Remove from Git history:**

```bash
# Install git-filter-repo (if not installed)
# macOS:
brew install git-filter-repo

# Remove .env from entire history
git filter-repo --path .env --invert-paths

# Force push (this rewrites history!)
git push origin --force --all
```

3. **Generate new API keys** and add them to a new `.env` file (which is now properly gitignored)

---

## Troubleshooting

### Issue: "Permission denied (publickey)"

**Solution:** Set up SSH keys or use HTTPS with personal access token

For HTTPS with token:
1. Go to GitHub Settings → Developer settings → Personal access tokens
2. Generate new token with `repo` permissions
3. Use token as password when prompted during `git push`

### Issue: "Repository not found"

**Solution:** Verify repository URL:
```bash
git remote -v
```

Update if needed:
```bash
git remote set-url origin https://github.com/savitamoorthi/Clinical_Intelligence_Assistant.git
```

### Issue: Merge conflicts

**Solution:**
```bash
# Pull with rebase
git pull --rebase origin main

# Fix conflicts in files, then:
git add <conflicted-files>
git rebase --continue
```

---

## Adding Collaborators (Optional)

If you want to add collaborators to your repository:

1. Go to your repository on GitHub
2. Click **Settings** → **Collaborators**
3. Click **Add people**
4. Enter their GitHub username or email

---

## Making Repository Public/Private

To change repository visibility:

1. Go to your repository on GitHub
2. Click **Settings**
3. Scroll to **Danger Zone**
4. Click **Change visibility**

---

## Additional Git Resources

- [GitHub Documentation](https://docs.github.com/)
- [Git Cheat Sheet](https://training.github.com/downloads/github-git-cheat-sheet/)
- [Pro Git Book (Free)](https://git-scm.com/book/en/v2)

---

**Created for:** Clinical Intelligence Assistant Project  
**Author:** Savita Moorthi  
**Last Updated:** December 2025
