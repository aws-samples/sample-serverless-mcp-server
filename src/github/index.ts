#!/usr/bin/env node
import middy from "@middy/core";
import httpErrorHandler from "@middy/http-error-handler";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";

import mcpMiddleware from "middy-mcp";
import cors from '@middy/http-cors'



import fetch from 'node-fetch';

import * as repository from './operations/repository.js';
import * as files from './operations/files.js';
import * as issues from './operations/issues.js';
import * as pulls from './operations/pulls.js';
import * as branches from './operations/branches.js';
import * as search from './operations/search.js';
import * as commits from './operations/commits.js';
import {
  GitHubError,
  GitHubValidationError,
  GitHubResourceNotFoundError,
  GitHubAuthenticationError,
  GitHubPermissionError,
  GitHubRateLimitError,
  GitHubConflictError,
  isGitHubError,
} from './common/errors.js';
import { VERSION } from "./common/version.js";
import { CreateIssueSchema, GetIssueSchema, IssueCommentSchema, ListIssuesOptionsSchema } from "./operations/issues.js";
import { SearchCodeSchema, SearchIssuesSchema, SearchUsersSchema } from "./operations/search.js";
import { CreateOrUpdateFileSchema, GetFileContentsSchema, PushFilesSchema } from "./operations/files.js";
import { CreateRepositoryOptionsSchema, SearchRepositoriesSchema } from "./operations/repository.js";
import { CreatePullRequestReviewSchema, CreatePullRequestSchema, GetPullRequestCommentsSchema, GetPullRequestFilesSchema, GetPullRequestReviewsSchema, GetPullRequestSchema, GetPullRequestStatusSchema, ListPullRequestsSchema, MergePullRequestSchema, UpdatePullRequestBranchSchema } from "./operations/pulls.js";

// If fetch doesn't exist in global scope, add it
if (!globalThis.fetch) {
  globalThis.fetch = fetch as unknown as typeof global.fetch;
}

const server = new McpServer({
  name: "Lambda hosted github-mcp-server",
  version: "1.0.0",
});


// Replace the ListToolsRequestSchema handler with individual tool registrations
server.tool("create_or_update_file", CreateOrUpdateFileSchema.shape, async (args) => {
  const result = await files.createOrUpdateFile(
    args.owner,
    args.repo,
    args.path,
    args.content,
    args.message,
    args.branch ?? "main",
    args.sha
  );
  return { content: [{ type: "text", text: JSON.stringify(result, null, 2) }] };
});


server.tool("search_repositories", SearchRepositoriesSchema.shape, async (args) => {
  const results = await repository.searchRepositories(
    args.query,
    args.page,
    args.perPage
  );
  return { content: [{ type: "text", text: JSON.stringify(results, null, 2) }] };
});

server.tool("create_repository", CreateRepositoryOptionsSchema.shape, async (args) => {
  const result = await repository.createRepository(args);
  return { content: [{ type: "text", text: JSON.stringify(result, null, 2) }] };
});

server.tool("get_file_contents", GetFileContentsSchema.shape, async (args) => {
  const contents = await files.getFileContents(
    args.owner,
    args.repo,
    args.path,
    args.branch
  );
  return { content: [{ type: "text", text: JSON.stringify(contents, null, 2) }] };
});

server.tool("push_files", PushFilesSchema.shape, async (args) => {
  const result = await files.pushFiles(
    args.owner,
    args.repo,
    args.branch,
    args.files,
    args.message
  );
  return { content: [{ type: "text", text: JSON.stringify(result, null, 2) }] };
});

server.tool("create_issue", CreateIssueSchema.shape, async (args) => {
  const { owner, repo, ...options } = args;
  try {
    console.error(`[DEBUG] Attempting to create issue in ${owner}/${repo}`);
    console.error(`[DEBUG] Issue options:`, JSON.stringify(options, null, 2));
    
    const issue = await issues.createIssue(owner, repo, options);
    
    console.error(`[DEBUG] Issue created successfully`);
    return { content: [{ type: "text", text: JSON.stringify(issue, null, 2) }] };
  } catch (err) {
    const error = err instanceof Error ? err : new Error(String(err));
    console.error(`[ERROR] Failed to create issue:`, error);
    
    if (error instanceof GitHubResourceNotFoundError) {
      throw new Error(
        `Repository '${owner}/${repo}' not found. Please verify:\n` +
        `1. The repository exists\n` +
        `2. You have correct access permissions\n` +
        `3. The owner and repository names are spelled correctly`
      );
    }
    
    throw new Error(
      `Failed to create issue: ${error.message}${
        error.stack ? `\nStack: ${error.stack}` : ''
      }`
    );
  }
});

server.tool("create_pull_request", CreatePullRequestSchema.shape, async (args) => {
  const pullRequest = await pulls.createPullRequest(args);
  return { content: [{ type: "text", text: JSON.stringify(pullRequest, null, 2) }] };
});

server.tool("search_code",SearchCodeSchema.shape, async (args) => {
  const results = await search.searchCode(args);
  return { content: [{ type: "text", text: JSON.stringify(results, null, 2) }] };
});

server.tool("search_issues", SearchIssuesSchema.shape, async (args) => {
  const results = await search.searchIssues(args);
  return { content: [{ type: "text", text: JSON.stringify(results, null, 2) }] };
});

server.tool("search_users", SearchUsersSchema.shape, async (args) => {
  const results = await search.searchUsers(args);
  return { content: [{ type: "text", text: JSON.stringify(results, null, 2) }] };
});


server.tool("list_issues", ListIssuesOptionsSchema.shape, async (args) => {
  const { owner, repo, ...options } = args;
  const result = await issues.listIssues(owner, repo, options);
  return { content: [{ type: "text", text: JSON.stringify(result, null, 2) }] };
});

server.tool("update_issue", issues.UpdateIssueOptionsSchema.shape, async (args) => {
  const { owner, repo, issue_number, ...options } = args;
  const result = await issues.updateIssue(owner, repo, issue_number, options);
  return { content: [{ type: "text", text: JSON.stringify(result, null, 2) }] };
});

server.tool("add_issue_comment", IssueCommentSchema.shape, async (args) => {
  const { owner, repo, issue_number, body } = args;
  const result = await issues.addIssueComment(owner, repo, issue_number, body);
  return { content: [{ type: "text", text: JSON.stringify(result, null, 2) }] };
});

server.tool("list_commits", {
  owner: z.string(),
  repo: z.string(),
  sha: z.string().optional(),
  path: z.string().optional(),
  author: z.string().optional(),
  since: z.string().optional(),
  until: z.string().optional(),
  page: z.number().optional(),
  perPage: z.number().optional()
}, async (args) => {
  const results = await commits.listCommits(
    args.owner,
    args.repo,
    args.page,
    args.perPage,
    args.sha
  );
  return { content: [{ type: "text", text: JSON.stringify(results, null, 2) }] };
});

server.tool("get_issue", GetIssueSchema.shape, async (args) => {
  const issue = await issues.getIssue(args.owner, args.repo, args.issue_number);
  return { content: [{ type: "text", text: JSON.stringify(issue, null, 2) }] };
});

server.tool("get_pull_request", GetPullRequestSchema.shape, async (args) => {
  const pullRequest = await pulls.getPullRequest(args.owner, args.repo, args.pull_number);
  return { content: [{ type: "text", text: JSON.stringify(pullRequest, null, 2) }] };
});

server.tool("list_pull_requests", ListPullRequestsSchema.shape, async (args) => {
  const { owner, repo, ...options } = args;
  const pullRequests = await pulls.listPullRequests(owner, repo, options);
  return { content: [{ type: "text", text: JSON.stringify(pullRequests, null, 2) }] };
});

server.tool("create_pull_request_review", CreatePullRequestReviewSchema.shape, async (args) => {
  const { owner, repo, pull_number, ...options } = args;
  const review = await pulls.createPullRequestReview(owner, repo, pull_number, options);
  return { content: [{ type: "text", text: JSON.stringify(review, null, 2) }] };
});

server.tool("merge_pull_request", MergePullRequestSchema.shape, async (args) => {
  const { owner, repo, pull_number, ...options } = args;
  const result = await pulls.mergePullRequest(owner, repo, pull_number, options);
  return { content: [{ type: "text", text: JSON.stringify(result, null, 2) }] };
});

server.tool("get_pull_request_files", GetPullRequestFilesSchema.shape, async (args) => {
  const files = await pulls.getPullRequestFiles(args.owner, args.repo, args.pull_number);
  return { content: [{ type: "text", text: JSON.stringify(files, null, 2) }] };
});

server.tool("get_pull_request_status", GetPullRequestStatusSchema.shape, async (args) => {
  const status = await pulls.getPullRequestStatus(args.owner, args.repo, args.pull_number);
  return { content: [{ type: "text", text: JSON.stringify(status, null, 2) }] };
});

server.tool("update_pull_request_branch", UpdatePullRequestBranchSchema.shape, async (args) => {
  const { owner, repo, pull_number, expected_head_sha } = args;
  await pulls.updatePullRequestBranch(owner, repo, pull_number, expected_head_sha);
  return { content: [{ type: "text", text: JSON.stringify({ success: true }, null, 2) }] };
});

server.tool("get_pull_request_comments",GetPullRequestCommentsSchema.shape, async (args) => {
  const comments = await pulls.getPullRequestComments(args.owner, args.repo, args.pull_number);
  return { content: [{ type: "text", text: JSON.stringify(comments, null, 2) }] };
});

server.tool("get_pull_request_reviews",GetPullRequestReviewsSchema.shape, async (args) => {
  const reviews = await pulls.getPullRequestReviews(args.owner, args.repo, args.pull_number);
  return { content: [{ type: "text", text: JSON.stringify(reviews, null, 2) }] };
});



export const handler = middy()
    .use(mcpMiddleware({ server }))
    .use(cors())
    .use(httpErrorHandler())
    .use({
      after: (request) => {
        const headers = { 
          "access-control-allow-headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
          "access-control-allow-methods": "GET,POST,PUT,DELETE,OPTIONS",
          "access-control-allow-origin": "*"
        };
        request.response = {
          ...request.response,
          headers: { ...request.response?.headers, ...headers },
        };
      },
    })
    
