create table if not exists sessions (
  id text primary key,
  user_id text,
  status text not null,
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

create table if not exists checkpoints (
  id bigserial primary key,
  session_id text not null references sessions(id),
  node_name text not null,
  state_blob jsonb not null,
  seq int not null,
  created_at timestamptz default now()
);

create table if not exists artifacts (
  id bigserial primary key,
  session_id text not null references sessions(id),
  type text not null,
  path text not null,
  checksum text,
  created_at timestamptz default now()
);

create table if not exists kg_nodes (
  id bigserial primary key,
  session_id text not null references sessions(id),
  entity text not null,
  claim text,
  source_url text,
  vector_id text,
  checkpoint_seq int,
  created_at timestamptz default now()
);
