# Deploying to Vercel

## Prerequisites
- A [Vercel](https://vercel.com) account
- [Vercel CLI](https://vercel.com/download) installed (optional for local deployment)

## Steps to Deploy

### 1. Prepare Your Environment Variables
Make sure to add the following environment variables to your Vercel project:
- `GOOGLE_GEMINI_API_KEY`: Your Google Gemini API key
- `SECRET_KEY`: A secure random string for session management

### 2. Using Vercel Dashboard

1. Login to your Vercel account
2. Click "New Project"
3. Import your GitHub repository
4. Configure the project:
   - Framework Preset: Other
   - Root Directory: ./
   - Build Command: None
   - Output Directory: None
5. Add the environment variables mentioned above
6. Click "Deploy"

### 3. Using Vercel CLI

1. Open terminal in your project directory
2. Login to Vercel:
   ```
   vercel login
   ```
3. Deploy:
   ```
   vercel
   ```
4. Follow prompts and make sure to add the required environment variables

### 4. Important Notes

- Serverless functions on Vercel have a maximum execution time of 10 seconds
- The filesystem is read-only except for `/tmp` directory
- Some features like long-running SSE might need adaptation for serverless environment

### 5. Testing Your Deployment

After deploying, visit your Vercel URL to verify the application works correctly.

### 6. Troubleshooting

If you encounter issues:
1. Check Vercel logs in the dashboard
2. Verify environment variables are set correctly
3. Make sure all dependencies are specified in requirements.txt
