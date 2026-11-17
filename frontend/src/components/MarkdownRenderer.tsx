'use client';

import React from 'react';

interface MarkdownRendererProps {
  content: string;
}

export default function MarkdownRenderer({ content }: MarkdownRendererProps) {
  if (!content) return null;

  // Split content into blocks by double newlines or section boundaries
  const lines = content.split('\n');
  const elements: React.ReactNode[] = [];

  let inList = false;
  let listItems: string[] = [];
  let listType: 'ul' | 'ol' = 'ul';

  let inTable = false;
  let tableHeader: string[] = [];
  let tableRows: string[][] = [];

  let inBlockquote = false;
  let blockquoteLines: string[] = [];

  const flushList = (key: string) => {
    if (listItems.length > 0) {
      if (listType === 'ol') {
        elements.push(
          <ol key={key} className="space-y-1.5 my-2.5 ml-4 list-decimal list-outside text-slate-200 text-sm">
            {listItems.map((item, idx) => (
              <li key={idx} className="leading-relaxed">
                {renderInlineFormatting(item)}
              </li>
            ))}
          </ol>
        );
      } else {
        elements.push(
          <ul key={key} className="space-y-1.5 my-2.5 ml-1 list-none text-slate-200 text-sm">
            {listItems.map((item, idx) => (
              <li key={idx} className="flex items-start gap-2 leading-relaxed">
                <span className="text-blue-400 font-bold shrink-0 mt-0.5">•</span>
                <span className="flex-1">{renderInlineFormatting(item)}</span>
              </li>
            ))}
          </ul>
        );
      }
      listItems = [];
      inList = false;
    }
  };

  const flushTable = (key: string) => {
    if (tableHeader.length > 0 || tableRows.length > 0) {
      elements.push(
        <div key={key} className="my-3 overflow-x-auto rounded-xl border border-slate-800 bg-slate-950/60">
          <table className="w-full text-left text-xs border-collapse">
            {tableHeader.length > 0 && (
              <thead className="bg-slate-900/90 text-slate-300 font-semibold border-b border-slate-800">
                <tr>
                  {tableHeader.map((th, idx) => (
                    <th key={idx} className="px-3.5 py-2.5 whitespace-nowrap">
                      {renderInlineFormatting(th)}
                    </th>
                  ))}
                </tr>
              </thead>
            )}
            <tbody className="divide-y divide-slate-800/60 text-slate-300">
              {tableRows.map((row, rIdx) => (
                <tr key={rIdx} className="hover:bg-slate-800/30 transition-colors">
                  {row.map((cell, cIdx) => (
                    <td key={cIdx} className="px-3.5 py-2 whitespace-nowrap font-mono text-[11px]">
                      {renderInlineFormatting(cell)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );
      tableHeader = [];
      tableRows = [];
      inTable = false;
    }
  };

  const flushBlockquote = (key: string) => {
    if (blockquoteLines.length > 0) {
      elements.push(
        <blockquote
          key={key}
          className="my-3 pl-4 py-2 border-l-2 border-blue-500/70 bg-blue-950/20 text-slate-300 italic text-xs rounded-r-xl leading-relaxed"
        >
          {blockquoteLines.map((line, idx) => (
            <p key={idx} className={idx > 0 ? 'mt-1' : ''}>
              {renderInlineFormatting(line)}
            </p>
          ))}
        </blockquote>
      );
      blockquoteLines = [];
      inBlockquote = false;
    }
  };

  for (let i = 0; i < lines.length; i++) {
    const rawLine = lines[i];
    const line = rawLine.trim();

    // Check table row
    if (line.startsWith('|') && line.endsWith('|')) {
      if (inList) flushList(`list-${i}`);
      if (inBlockquote) flushBlockquote(`quote-${i}`);

      const parts = line.split('|').map(p => p.trim()).filter((_, idx, arr) => idx > 0 && idx < arr.length - 1);
      
      // Check if separator line (|---|---|)
      const isSeparator = parts.every(p => /^:?-+:?$/.test(p));
      if (isSeparator) {
        continue;
      }

      if (!inTable) {
        inTable = true;
        tableHeader = parts;
      } else {
        tableRows.push(parts);
      }
      continue;
    } else if (inTable) {
      flushTable(`table-${i}`);
    }

    // Check blockquote
    if (line.startsWith('>')) {
      if (inList) flushList(`list-${i}`);
      inBlockquote = true;
      blockquoteLines.push(line.replace(/^>\s?/, ''));
      continue;
    } else if (inBlockquote) {
      flushBlockquote(`quote-${i}`);
    }

    // Check unordered list
    if (line.startsWith('- ') || line.startsWith('* ')) {
      if (!inList || listType !== 'ul') {
        flushList(`list-switch-${i}`);
        inList = true;
        listType = 'ul';
      }
      listItems.push(line.slice(2).trim());
      continue;
    }

    // Check ordered list
    const olMatch = line.match(/^(\d+)[\.\)]\s+(.+)$/);
    if (olMatch) {
      if (!inList || listType !== 'ol') {
        flushList(`list-switch-${i}`);
        inList = true;
        listType = 'ol';
      }
      listItems.push(olMatch[2].trim());
      continue;
    }

    // If we were in a list and hit non-list text
    if (inList) {
      flushList(`list-${i}`);
    }

    // Empty line
    if (!line) {
      continue;
    }

    // Headers
    if (line.startsWith('### ')) {
      elements.push(
        <h3 key={`h3-${i}`} className="text-base font-bold text-slate-100 mt-4 mb-2 flex items-center gap-2">
          {renderInlineFormatting(line.slice(4))}
        </h3>
      );
      continue;
    }

    if (line.startsWith('#### ')) {
      elements.push(
        <h4 key={`h4-${i}`} className="text-sm font-semibold text-slate-200 mt-3 mb-1.5 uppercase tracking-wide text-blue-300/90 text-xs">
          {renderInlineFormatting(line.slice(5))}
        </h4>
      );
      continue;
    }

    if (line.startsWith('## ')) {
      elements.push(
        <h2 key={`h2-${i}`} className="text-lg font-extrabold text-white mt-5 mb-2.5 pb-1 border-b border-slate-800">
          {renderInlineFormatting(line.slice(3))}
        </h2>
      );
      continue;
    }

    if (line.startsWith('# ')) {
      elements.push(
        <h1 key={`h1-${i}`} className="text-xl font-extrabold text-white mt-6 mb-3">
          {renderInlineFormatting(line.slice(2))}
        </h1>
      );
      continue;
    }

    // Horizontal rule
    if (/^---+$|^\*\*\*+$/.test(line)) {
      elements.push(<hr key={`hr-${i}`} className="my-3 border-slate-800" />);
      continue;
    }

    // Regular paragraph
    elements.push(
      <p key={`p-${i}`} className="text-slate-200 text-sm leading-relaxed my-2">
        {renderInlineFormatting(rawLine)}
      </p>
    );
  }

  // Flush any remaining active containers
  if (inList) flushList('final-list');
  if (inTable) flushTable('final-table');
  if (inBlockquote) flushBlockquote('final-quote');

  return <div className="space-y-1 text-sm">{elements}</div>;
}

function renderInlineFormatting(text: string): React.ReactNode {
  if (!text) return null;

  // Split by bold (**bold**), inline code (`code`), or italic (*italic*)
  const tokens = [];
  let remaining = text;

  // Regular expression to match bold (**...**), code (`...`), or italic (*...*)
  const inlineRegex = /(\*\*([^*]+)\*\*|`([^`]+)`|\*([^*]+)\*)/g;
  let match;
  let lastIndex = 0;
  let keyIdx = 0;

  while ((match = inlineRegex.exec(text)) !== null) {
    const matchStart = match.index;
    const matchEnd = match.index + match[0].length;

    // Plain text before match
    if (matchStart > lastIndex) {
      tokens.push(text.slice(lastIndex, matchStart));
    }

    if (match[2]) {
      // Bold
      tokens.push(
        <strong key={`b-${keyIdx++}`} className="font-semibold text-slate-100">
          {match[2]}
        </strong>
      );
    } else if (match[3]) {
      // Code
      tokens.push(
        <code key={`c-${keyIdx++}`} className="px-1.5 py-0.5 rounded bg-slate-950 border border-slate-800 font-mono text-[11px] text-cyan-300">
          {match[3]}
        </code>
      );
    } else if (match[4]) {
      // Italic
      tokens.push(
        <em key={`i-${keyIdx++}`} className="italic text-slate-300">
          {match[4]}
        </em>
      );
    }

    lastIndex = matchEnd;
  }

  if (lastIndex < text.length) {
    tokens.push(text.slice(lastIndex));
  }

  return tokens.length > 0 ? tokens : text;
}
