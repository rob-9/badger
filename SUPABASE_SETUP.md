# Supabase Setup Guide for Boom

## Step 1: Create Supabase Account

1. Go to [supabase.com](https://supabase.com)
2. Sign up for a free account (no credit card required)
3. Click "New Project"

## Step 2: Create a New Project

1. **Organization**: Create or select an organization
2. **Project Name**: `boom-waitlist` (or whatever you prefer)
3. **Database Password**: Create a strong password (save it somewhere safe)
4. **Region**: Choose the closest region to your users
5. Click "Create new project" and wait ~2 minutes for setup

## Step 3: Create the Waitlist Table

1. In your Supabase project, go to **Table Editor** (left sidebar)
2. Click "Create a new table"
3. Configure the table:
   - **Name**: `waitlist`
   - **Description**: Early access email signups
   - Enable "Enable Row Level Security (RLS)"

4. Add columns:
   - `id` (int8, primary key) - **Already created by default**
   - `created_at` (timestamptz, default: now()) - **Already created by default**
   - `email` (text, required, unique)
     - Click "Add column"
     - Name: `email`
     - Type: `text`
     - Check "Is Nullable": **No** (required)
     - Check "Is Unique": **Yes**
     - Click "Save"

5. Click "Save" to create the table

## Step 4: Configure Row Level Security (RLS)

Since we want anyone to be able to submit emails (INSERT), we need to add a policy:

1. In the **Table Editor**, click on the `waitlist` table
2. Click "Add RLS policy" or go to **Authentication > Policies**
3. Click "New Policy"
4. Select "Create a policy from scratch"
5. Configure:
   - **Policy name**: `Enable insert for anonymous users`
   - **Allowed operation**: SELECT `INSERT`
   - **Target roles**: `anon` (this is the anonymous public role)
   - **USING expression**: `true`
   - **WITH CHECK expression**: `true`
6. Click "Review" then "Save policy"

## Step 5: Get Your API Keys

1. Go to **Project Settings** (gear icon in sidebar)
2. Click **API** in the left menu
3. Copy these two values:
   - **Project URL** (looks like: `https://xxxxxxxxxxxxx.supabase.co`)
   - **anon public** key (starts with `eyJ...`)

## Step 6: Add Keys to Your .env.local File

1. Open `/Users/robert/curio/.env.local`
2. Replace the placeholder values:
   ```
   NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
   NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
   ```
3. Save the file
4. **Restart your Next.js dev server** for the env vars to take effect

## Step 7: Test It!

1. Go to `http://localhost:3000/early-access`
2. Enter an email and submit
3. Check your Supabase dashboard under **Table Editor > waitlist** to see the new entry!

## Viewing Your Waitlist

You can view all collected emails in the Supabase dashboard:

1. Go to **Table Editor**
2. Click on `waitlist` table
3. You'll see all submitted emails with timestamps

You can also export to CSV:
1. Click the table
2. Click the "..." menu
3. Select "Download as CSV"

## Optional: Email Notifications

To get notified when someone joins the waitlist, you can:

1. Set up a Database Webhook in Supabase
2. Use a service like [Zapier](https://zapier.com) or [n8n](https://n8n.io)
3. Or create a Supabase Edge Function to send emails

---

## Files Created/Modified

- ✅ `/src/lib/supabase.ts` - Supabase client configuration
- ✅ `/src/app/early-access/page.tsx` - Updated to save emails to Supabase
- ✅ `/.env.local` - Added Supabase environment variables
- ✅ Installed `@supabase/supabase-js` package

---

## Troubleshooting

**Error: "Invalid API key"**
- Make sure you copied the `anon public` key, not the `service_role` key
- Restart your dev server after updating `.env.local`

**Error: "relation 'waitlist' does not exist"**
- Make sure you created the table named exactly `waitlist` (lowercase)

**Error: "new row violates row-level security policy"**
- Make sure you created the RLS policy for INSERT operations
- Check that the policy is enabled

**Duplicate email error not showing**
- This is working correctly - duplicate emails will show "This email is already on the waitlist"
