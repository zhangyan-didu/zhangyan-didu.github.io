import type {ReactNode} from 'react';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

const ProjectList = [
  {
    title: '个人博客系统',
    description: '基于 Docusaurus 构建的个人博客，记录技术学习和生活感悟',
    technologies: ['React', 'TypeScript', 'Docusaurus'],
    date: '2024-09-15',
    status: '已完成',
    link: '#',
  },
  {
    title: '任务管理应用',
    description: '一款简洁优雅的任务管理工具，帮助提高工作效率',
    technologies: ['Vue.js', 'Node.js', 'MongoDB'],
    date: '2024-08-20',
    status: '开发中',
    link: '#',
  },
  {
    title: '数据可视化平台',
    description: '将复杂数据转化为直观的图表和仪表板',
    technologies: ['React', 'D3.js', 'Python'],
    date: '2024-07-10',
    status: '已完成',
    link: '#',
  },
  {
    title: '移动端天气应用',
    description: '响应式天气应用，提供准确的天气信息和预报',
    technologies: ['React Native', 'Weather API'],
    date: '2024-06-15',
    status: '计划中',
    link: '#',
  },
];

const getStatusColor = (status: string) => {
  switch(status) {
    case '已完成':
      return '#4caf50';
    case '开发中':
      return '#ff9800';
    case '计划中':
      return '#2196f3';
    default:
      return '#666';
  }
};

export default function ProjectsPage(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title="Project Display"
      description="展示个人作品和项目经验">
      <div className={styles.container}>
        <div className={styles.header}>
          <Heading as="h1">Project Display</Heading>
          <p className={styles.subtitle}>项目展示 · 从构思到实现，完整记录开发过程和技术亮点</p>
        </div>

        <div className={styles.grid}>
          {ProjectList.map((project, index) => (
            <div key={index} className={styles.card}>
              <div className={styles.cardContent}>
                <div className={styles.headerRow}>
                  <Heading as="h3">{project.title}</Heading>
                  <span
                    className={styles.status}
                    style={{ backgroundColor: getStatusColor(project.status) }}
                  >
                    {project.status}
                  </span>
                </div>
                <p className={styles.description}>{project.description}</p>

                <div className={styles.techStack}>
                  <strong>技术栈：</strong>
                  <div className={styles.techTags}>
                    {project.technologies.map((tech, techIndex) => (
                      <span key={techIndex} className={styles.techTag}>
                        {tech}
                      </span>
                    ))}
                  </div>
                </div>

                <div className={styles.date}>{project.date}</div>

                <Link className="button button--secondary" to={project.link}>
                  查看详情
                </Link>
              </div>
            </div>
          ))}
        </div>
      </div>
    </Layout>
  );
}