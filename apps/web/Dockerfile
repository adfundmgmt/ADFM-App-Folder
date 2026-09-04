FROM node:22-alpine AS deps
WORKDIR /app
COPY apps/web/package.json ./package.json
RUN npm install

FROM node:22-alpine AS builder
WORKDIR /app
ARG NEXT_PUBLIC_ADFM_API_URL=http://localhost:8000
ENV NEXT_PUBLIC_ADFM_API_URL=$NEXT_PUBLIC_ADFM_API_URL
COPY --from=deps /app/node_modules ./node_modules
COPY apps/web ./
RUN npm run build

FROM node:22-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./package.json
COPY --from=builder /app/public ./public
EXPOSE 3000
CMD ["npm", "run", "start"]
