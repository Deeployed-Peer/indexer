-- CreateEnum
CREATE TYPE "GraphBuildStatus" AS ENUM ('PENDING', 'COMPLETED', 'FAILED');

-- CreateEnum
CREATE TYPE "SymbolKind" AS ENUM ('MODULE', 'CLASS', 'INTERFACE', 'FUNCTION', 'METHOD', 'PROPERTY', 'VARIABLE', 'UNKNOWN');

-- CreateEnum
CREATE TYPE "SymbolRole" AS ENUM ('DEFINITION', 'REFERENCE');

-- CreateEnum
CREATE TYPE "EdgeType" AS ENUM ('CONTAINS', 'INVOKES', 'REFERENCES', 'IMPLEMENTS', 'EXTENDS', 'IMPORTS');

-- CreateEnum
CREATE TYPE "TagKind" AS ENUM ('DEF', 'REF');

-- CreateEnum
CREATE TYPE "TagCategory" AS ENUM ('CLASS', 'FUNCTION');

-- CreateTable
CREATE TABLE "GraphBuild" (
    "id" TEXT NOT NULL,
    "repositoryUrl" TEXT NOT NULL,
    "provider" TEXT NOT NULL,
    "branch" TEXT,
    "commitSha" TEXT,
    "storageKey" TEXT,
    "graphByteSize" INTEGER,
    "status" "GraphBuildStatus" NOT NULL DEFAULT 'PENDING',
    "errorMessage" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "completedAt" TIMESTAMP(3),

    CONSTRAINT "GraphBuild_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "RepoFile" (
    "id" TEXT NOT NULL,
    "graphBuildId" TEXT NOT NULL,
    "relativePath" TEXT NOT NULL,
    "language" TEXT NOT NULL,
    "contentHash" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "RepoFile_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "Symbol" (
    "id" TEXT NOT NULL,
    "graphBuildId" TEXT NOT NULL,
    "fileId" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "role" "SymbolRole" NOT NULL,
    "kind" "SymbolKind" NOT NULL,
    "startLine" INTEGER NOT NULL,
    "endLine" INTEGER NOT NULL,
    "containerId" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "Symbol_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "SymbolEdge" (
    "id" TEXT NOT NULL,
    "graphBuildId" TEXT NOT NULL,
    "sourceSymbolId" TEXT NOT NULL,
    "targetSymbolId" TEXT NOT NULL,
    "edgeType" "EdgeType" NOT NULL,

    CONSTRAINT "SymbolEdge_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "CodeTag" (
    "id" TEXT NOT NULL,
    "graphBuildId" TEXT NOT NULL,
    "fileId" TEXT,
    "symbolId" TEXT,
    "rel_fname" TEXT NOT NULL,
    "fname" TEXT NOT NULL,
    "kind" "TagKind" NOT NULL,
    "category" "TagCategory" NOT NULL,
    "info" TEXT NOT NULL,
    "lineStart" INTEGER NOT NULL,
    "lineEnd" INTEGER NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "CodeTag_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "RepoFile_graphBuildId_relativePath_idx" ON "RepoFile"("graphBuildId", "relativePath");

-- CreateIndex
CREATE INDEX "Symbol_graphBuildId_name_idx" ON "Symbol"("graphBuildId", "name");

-- CreateIndex
CREATE INDEX "Symbol_fileId_startLine_endLine_idx" ON "Symbol"("fileId", "startLine", "endLine");

-- CreateIndex
CREATE INDEX "SymbolEdge_graphBuildId_sourceSymbolId_edgeType_idx" ON "SymbolEdge"("graphBuildId", "sourceSymbolId", "edgeType");

-- CreateIndex
CREATE INDEX "SymbolEdge_graphBuildId_targetSymbolId_edgeType_idx" ON "SymbolEdge"("graphBuildId", "targetSymbolId", "edgeType");

-- CreateIndex
CREATE INDEX "CodeTag_graphBuildId_rel_fname_kind_idx" ON "CodeTag"("graphBuildId", "rel_fname", "kind");

-- AddForeignKey
ALTER TABLE "RepoFile" ADD CONSTRAINT "RepoFile_graphBuildId_fkey" FOREIGN KEY ("graphBuildId") REFERENCES "GraphBuild"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Symbol" ADD CONSTRAINT "Symbol_graphBuildId_fkey" FOREIGN KEY ("graphBuildId") REFERENCES "GraphBuild"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Symbol" ADD CONSTRAINT "Symbol_fileId_fkey" FOREIGN KEY ("fileId") REFERENCES "RepoFile"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "SymbolEdge" ADD CONSTRAINT "SymbolEdge_graphBuildId_fkey" FOREIGN KEY ("graphBuildId") REFERENCES "GraphBuild"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "SymbolEdge" ADD CONSTRAINT "SymbolEdge_sourceSymbolId_fkey" FOREIGN KEY ("sourceSymbolId") REFERENCES "Symbol"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "SymbolEdge" ADD CONSTRAINT "SymbolEdge_targetSymbolId_fkey" FOREIGN KEY ("targetSymbolId") REFERENCES "Symbol"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "CodeTag" ADD CONSTRAINT "CodeTag_graphBuildId_fkey" FOREIGN KEY ("graphBuildId") REFERENCES "GraphBuild"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "CodeTag" ADD CONSTRAINT "CodeTag_fileId_fkey" FOREIGN KEY ("fileId") REFERENCES "RepoFile"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "CodeTag" ADD CONSTRAINT "CodeTag_symbolId_fkey" FOREIGN KEY ("symbolId") REFERENCES "Symbol"("id") ON DELETE SET NULL ON UPDATE CASCADE;
